"""
Pruebas del módulo skill_shield/teleop_fr5 (sin hardware ni MuJoCo):

    python -m pytest tests_teleop_fr5.py -q          # o simplemente:  python tests_teleop_fr5.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_shield.teleop_fr5.cinematica import CadenaFR5, OFFSET_MUNECA_ORCA, T_de, cargar_cadena
from skill_shield.teleop_fr5.ik_teleop import ControlIKFR5, ModeloColision, ParametrosIK, entorno_soporte_hombros
from skill_shield.teleop_fr5.retargeting import Retargeting, ConfigRetargeting

RUTA_MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fairino_description", "mjcf", "fairino_mjcf.xml")


def _cadena():
    return CadenaFR5.desde_mjcf(RUTA_MJCF) if os.path.exists(RUTA_MJCF) else CadenaFR5.desde_dh()


def test_fk_equivale_a_dh():
    c = _cadena()
    dh = c.dh_equivalente()
    ref = CadenaFR5.desde_dh(params=dh["params"], T_base=c.T_base @ dh["T_0"], T_flange_tool=np.eye(4))
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(c.limites[:, 0], c.limites[:, 1])
        Ta = c.fk_marcos(q)[6]
        Tb = ref.fk_marcos(q * dh["signos"])[6] @ dh["T_6tool"]
        assert np.abs(Ta - Tb).max() < 1e-9


def test_ik_analitica_ida_y_vuelta():
    c = _cadena()
    rng = np.random.default_rng(2)
    recuperadas = 0
    for _ in range(300):
        q = rng.uniform(c.limites[:, 0], c.limites[:, 1])
        T = c.fk_marcos(q)[6]
        sols = c.ik_analitica(T, q_ref=q)
        assert sols, "sin soluciones para una pose alcanzable"
        for s in sols:
            Ts = c.fk_marcos(s)[6]
            assert np.linalg.norm(Ts[:3, 3] - T[:3, 3]) < 1e-6
            assert np.abs(Ts[:3, :3] - T[:3, :3]).max() < 1e-6
        if min(np.abs(np.asarray(s) - q).max() for s in sols) < 1e-6:
            recuperadas += 1
    assert recuperadas == 300


def test_jacobiana_numerica():
    c = _cadena()
    q = np.array([0.3, -1.1, 1.4, -0.7, 1.2, 0.5])
    Jl, Ja = c.jacobiana(q, OFFSET_MUNECA_ORCA)
    p0 = c.punto(q, OFFSET_MUNECA_ORCA)
    h = 1e-6
    for i in range(6):
        dq = np.zeros(6); dq[i] = h
        num = (c.punto(q + dq, OFFSET_MUNECA_ORCA) - p0) / h
        assert np.abs(num - Jl[:, i]).max() < 1e-4


def test_fusion_robusta_a_oclusion():
    from skill_shield.teleop_fr5.simulador import simular_fusion
    resumen, serie = simular_fusion(verbose=False)
    for f, r in resumen.items():
        assert r["ok"] == r["n"], f"fase {f}: fotogramas sin salida"
        assert r["eW_med"] < 0.02, f"fase {f}: muñeca med {r['eW_med']:.3f} m"
        assert r["eE_med"] < 0.02, f"fase {f}: codo med {r['eE_med']:.3f} m"
        assert r["eW_p95"] < 0.04, f"fase {f}: muñeca p95 {r['eW_p95']:.3f} m"
        assert r["eE_p95"] < 0.06, f"fase {f}: codo p95 {r['eE_p95']:.3f} m"
    # la fase 2 (mano hacia la cámara A todo el rato) es la que rompía la fusión antigua
    assert resumen[2]["eE_ing_p95"] > resumen[2]["eE_p95"] * 3


def test_teleop_sin_colisiones_ni_limites():
    from skill_shield.teleop_fr5.simulador import simular_teleop
    res, serie = simular_teleop(verbose=False)
    assert res["fuera_limites"] == 0
    assert res["colisiones"] == 0
    assert res["v_max"] <= 3.0 + 1e-6
    assert res["estados"].get("exacta", 0) + res["estados"].get("aprox", 0) > 380
    assert res["err_med"] < 0.005 and res["err_p95"] < 0.03
    # postura humana: codo de un lado, muñeca sin voltear, sin "vueltas"
    Q = np.array([x["q"] for x in serie if "q" in x])
    assert np.all(Q[:, 4] > 0) and np.all(np.abs(Q[:, 3]) <= math.radians(80) + 1e-6)
    assert np.all(Q[:, 2] >= -math.radians(5) - 1e-6)
    assert np.abs(np.diff(Q, axis=0)).max() < 0.2           # ningún salto articular entre fotogramas


def test_otro_camino_cambio_de_rama():
    """Objetivo alcanzable solo desde otra rama de la IK → transición planificada, sin
    colisiones, a velocidad limitada, que termina en la rama nueva con error ~0."""
    c = _cadena()
    inclinada = not np.allclose(c.T_base[:3, :3], np.eye(3))
    col = ModeloColision(c, entorno_soporte_hombros() if inclinada else None)
    par = ParametrosIK(limites_extra={2: (-2.53, 2.53)}, postura_humana=False)   # todas las ramas
    ctl = ControlIKFR5(c, col, par)
    rng = np.random.default_rng(5)

    def objetivo_de(q):
        fk = c.fk(q)
        return dict(p_muneca=fk["muneca"], R_mano=fk["T_tool"][:3, :3], p_codo_sugerido=fk["codo"])

    # arranque en una postura válida
    while True:
        q_a = rng.uniform(ctl.lo, ctl.hi)
        if ctl._valida(q_a)[0]:
            break
    ctl.q = q_a.copy()
    q, info = ctl.resolver(objetivo_de(q_a), 0.0, 0.04)
    assert info["estado"] == "exacta"
    rama_a = info["rama"]
    # buscar un objetivo cuyas soluciones válidas NO incluyan la rama actual y estén lejos
    encontrado = None
    for _ in range(2000):
        q_b = rng.uniform(ctl.lo, ctl.hi)
        if not ctl._valida(q_b)[0]:
            continue
        T = c.fk_marcos(q_b)[6]
        validas = [(s, r) for s, r in c.ik_analitica(T, q_ref=ctl.q, con_rama=True) if ctl._valida(s)[0]]
        if validas and all(r != rama_a for _, r in validas) and ctl._dist_q(validas[0][0], ctl.q) > 2.0:
            encontrado = q_b
            break
    assert encontrado is not None
    obj = objetivo_de(encontrado)
    t, dt = 0.0, 0.04
    estados = []
    for k in range(400):
        t += dt
        q, info = ctl.resolver(obj, t, dt)
        estados.append(info["estado"])
        assert ctl.en_limites(q)
        assert col.libre(q), f"colisión durante '{info['estado']}' en el paso {k}"
        if k > 0:
            assert np.abs(q - q_prev).max() <= max(par.v_transicion, par.v_seguimiento) * dt + 1e-9
        q_prev = q.copy()
        if info["estado"] == "exacta" and info.get("error", 1.0) < 1e-6:
            break
    assert "transicion" in estados, estados[:10]
    assert info["estado"] == "exacta" and info["rama"] != rama_a
    assert np.linalg.norm(c.punto(q, OFFSET_MUNECA_ORCA) - obj["p_muneca"]) < 1e-6


def test_postura_humana_descarta_munecas_volteadas():
    """Con la postura humana activa, ninguna solución elegida tiene la muñeca volteada ni el
    codo al revés, y las cajas articulares salen de los límites anatómicos."""
    c = _cadena()
    ctl = ControlIKFR5(c, ModeloColision(c), ParametrosIK())
    assert ctl.postura is not None and ctl.postura.codo_signo in (1, -1)
    assert math.degrees(ctl.hi[3]) <= 80.0 + 1e-6 and math.degrees(ctl.lo[4]) >= 30.0 - 1e-6
    rng = np.random.default_rng(9)
    n = 0
    for _ in range(300):
        q = rng.uniform(ctl.lo, ctl.hi)
        if not ctl._valida(q)[0]:
            continue
        fk = c.fk(q)
        obj = dict(p_muneca=fk["muneca"], R_mano=fk["T_tool"][:3, :3], p_codo_sugerido=fk["codo"])
        ctl.q, ctl.rama = None, None
        qs, info = ctl.resolver(obj, 0.0, None)
        assert qs[4] > 0 and abs(qs[3]) <= math.radians(80) + 1e-6
        assert ctl.postura.codo_signo * qs[2] >= -math.radians(5) - 1e-6
        n += 1
    assert n > 50


def test_objetivo_imposible_no_choca():
    """Muñeca pedida dentro de la columna del soporte / bajo el suelo: el robot se acerca lo
    que puede ('aprox' o 'bloqueado') sin chocar ni salirse de límites."""
    c = _cadena()
    if np.allclose(c.T_base[:3, :3], np.eye(3)):
        return                                    # el entorno de columna es del soporte inclinado
    col = ModeloColision(c, entorno_soporte_hombros())
    ctl = ControlIKFR5(c, col, ParametrosIK())
    q0 = np.array([0.0, -1.5708, 1.35, 0.0, 1.5708, 0.0])
    assert col.libre(q0)
    ctl.q = q0.copy()
    rt = Retargeting(c, ConfigRetargeting())
    t = 0.0
    for p_obj in (np.array([0.26, 0.0, -0.2]), np.array([-0.4, -0.3, -1.1]), np.array([-0.8, -0.9, 0.2])):
        R = rt.R_mano(np.array([0.0, -1.0, 0.0]), 0.0)
        for _ in range(60):
            t += 0.04
            q, info = ctl.resolver(dict(p_muneca=p_obj, R_mano=R), t, 0.04)
            assert ctl.en_limites(q)
            assert col.libre(q), info
        assert info["estado"] in ("aprox", "bloqueado", "exacta", "transicion")


if __name__ == "__main__":
    import inspect
    fallos = 0
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"✅ {nombre}")
            except AssertionError as e:
                fallos += 1
                print(f"❌ {nombre}: {e}")
    sys.exit(1 if fallos else 0)

"""
Pruebas del ARCTOS en el hombro izquierdo (29-oct), sin hardware ni MuJoCo:

    python -m pytest tests_teleop_arctos.py -q

Comprueban que el ARCTOS usa la misma teleoperación que el FR5 y que la sigue igual de bien
(diagnóstico de IK con la trayectoria limpia y con ruido), además de la cinemática, la mano
izquierda y las colisiones entre los dos brazos.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_shield.teleop_fr5.cinematica import (
    CadenaArctos, CadenaFR5, OFFSET_MUNECA_ORCA_IZQ, quat_a_mat, rot_eje, euler_xyz_a_mat, wrap,
)
from skill_shield.teleop_fr5.retargeting import Retargeting

RUTA_MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fairino_description", "mjcf", "fairino_mjcf.xml")


def _arctos():
    return CadenaArctos.desde_mjcf(RUTA_MJCF)


def _fk_xml(nombre_body, q):
    """FK independiente (recorriendo el XML) del body 'nombre_body' con los joints de q."""
    raiz = ET.parse(RUTA_MJCF).getroot()

    def R_de(b):
        if b.get("quat"):
            return quat_a_mat([float(v) for v in b.get("quat").split()])
        if b.get("euler"):
            return euler_xyz_a_mat([float(v) for v in b.get("euler").split()])
        return np.eye(3)

    def rec(el, R, p):
        for b in el.findall("body"):
            Rb = R @ R_de(b)
            pb = p + R @ np.array([float(v) for v in b.get("pos", "0 0 0").split()])
            j = b.find("joint")
            if j is not None and j.get("name") in q:
                Rb = Rb @ rot_eje([float(v) for v in j.get("axis").split()], q[j.get("name")])
            if b.get("name") == nombre_body:
                return Rb, pb
            r = rec(b, Rb, pb)
            if r is not None:
                return r
        return None
    return rec(raiz.find("worldbody"), np.eye(3), np.zeros(3))


def test_modelo_tiene_los_dos_brazos_y_el_orden_de_actuadores():
    raiz = ET.parse(RUTA_MJCF).getroot()
    acts = [a.get("name") for a in raiz.find("actuator")]
    assert acts[:6] == [f"act_joint{i}" for i in range(1, 7)]
    assert acts[6] == "right_wrist_actuator" and acts[22] == "right_t-pip_actuator"
    assert acts[23:29] == [f"act_arctos_joint{i}" for i in range(1, 7)]
    assert acts[29] == "left_wrist_actuator" and acts[45] == "left_t-pip_actuator"
    nombres = [e.get("name") for e in list(raiz.iter("body")) + list(raiz.iter("joint")) if e.get("name")]
    assert len(nombres) == len(set(nombres)), "nombres de body/joint repetidos"


def test_fk_cadena_igual_que_el_xml():
    c = _arctos()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(c.limites[:, 0], c.limites[:, 1])
        R, p = _fk_xml("mano_mount_izq", {f"arctos_joint{i + 1}": q[i] for i in range(6)})
        T = c.fk_marcos(q)[6]
        assert np.allclose(T[:3, 3], p, atol=1e-9) and np.allclose(T[:3, :3], R, atol=1e-9)


def test_arctos_en_el_hombro_izquierdo():
    c = _arctos()
    fk = c.fk(np.zeros(6))
    fr5 = CadenaFR5.desde_mjcf(RUTA_MJCF).fk(np.zeros(6))
    assert fk["hombro"][0] > fr5["hombro"][0] + 0.4          # a la izquierda del FR5 (+X)
    eje = fk["ejes"][0]
    assert np.allclose(eje, [math.sqrt(0.5), 0.0, -math.sqrt(0.5)], atol=1e-6)   # 45° fuera-abajo
    # con todos los joints a 0 el antebrazo apunta al frente (−Y)
    _, codo, mc, _ = c.puntos_brazo(fk["marcos"])
    f = (mc - codo) / np.linalg.norm(mc - codo)
    assert f[1] < -0.95


def test_ik_analitica_ida_y_vuelta():
    c = _arctos()
    rng = np.random.default_rng(2)
    n = ok = 0
    for _ in range(400):
        q = rng.uniform(c.limites[:, 0], c.limites[:, 1])
        if not (0.2 < c.codo_recto - q[2] < math.radians(140)) or not (0.3 < abs(q[4]) < 2.4):
            continue
        n += 1
        T = c.fk_marcos(q)[6]
        sols = c.ik_analitica(T, q_ref=q)
        for s in sols:                                     # todas llegan al objetivo (muñeca no esférica: mm)
            Ts = c.fk_marcos(s)[6]
            assert np.linalg.norm(Ts[:3, 3] - T[:3, 3]) < 6e-3 and np.allclose(Ts[:3, :3], T[:3, :3], atol=1e-6)
        if any(np.abs([wrap(a - b) for a, b in zip(s, q)]).max() < 5e-3 for s in sols):
            ok += 1
    assert n > 100 and ok >= 0.95 * n, f"{ok}/{n}"


def test_jacobiana_numerica_mano_izquierda():
    c = _arctos()
    q = np.array([0.3, 0.4, 0.8, -0.2, 0.6, 0.1])
    Jl, _ = c.jacobiana(q, OFFSET_MUNECA_ORCA_IZQ)
    p0 = c.punto(q, OFFSET_MUNECA_ORCA_IZQ)
    h = 1e-6
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = h
        assert np.allclose((c.punto(q + dq, OFFSET_MUNECA_ORCA_IZQ) - p0) / h, Jl[:, k], atol=1e-5)


def test_convencion_mano_izquierda():
    """R_mano de la izquierda es una rotación propia, dedos = antebrazo, y roll_desde_R la invierte."""
    r = Retargeting(_arctos())
    assert r.izquierda
    f = np.array([0.3, -0.8, -0.5])
    f /= np.linalg.norm(f)
    for roll in (-2.0, -0.4, 0.0, 1.1, 2.5):
        R = r.R_mano(f, roll)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0.999
        assert np.allclose(R[:, 1], f)
        assert abs(wrap(r.roll_desde_R(R) - roll)) < 1e-9


def _teleop(r, s, semilla=0):
    from skill_shield.teleop_fr5.diagnostico import construir_teleop, diagnosticar_ik
    t, q0, _ = construir_teleop(brazo="izquierdo", usar_config=False)
    return diagnosticar_ik(r, s, teleop=t, q0=q0, verbose=False, semilla=semilla)


def test_diagnostico_ik_limpio_igual_o_mejor_que_el_fr5():
    """Trayectoria limpia del diagnóstico (T-pose, colgando, arriba, frente, codo, cara...)."""
    res, serie = _teleop(0.0, 0.0)
    assert res["pct"].get("exacta", 0) > 85.0              # FR5: 80 %
    assert res["err_real_p95"] < 0.04                      # FR5: 0,33 m (hueco del hombro al colgar)
    assert res["n_transiciones"] == 0 and res["s_sin_seguir"] == 0
    ex = [x for x in serie if x["estado"] == "exacta" and x["fase"] != "arranque" and "err_real" in x]
    assert np.median([x["err_real"] for x in ex]) < 1e-3


def test_diagnostico_ik_con_ruido_sin_quedarse_atascado():
    """1 cm de ruido + saltos de 3 cm (como al cambiar de cámara): en varias semillas, nunca se
    queda con el codo al revés (la semilla 3 lo hacía al pasar por el eje de joint1)."""
    for semilla in range(4):
        res, _ = _teleop(1.0, 3.0, semilla)
        assert res["pct"].get("exacta", 0) > 75.0, (semilla, res["pct"])
        assert res["err_real_p95"] < 0.09 and res["ang_real_p95"] < 25.0, semilla


def test_colisiones_entre_brazos():
    from skill_shield.teleop_fr5.brazo_arctos import compartir_obstaculos
    from skill_shield.teleop_fr5.diagnostico import construir_teleop
    from skill_shield.teleop_fr5.ik_teleop import Capsula
    der, q_der, _ = construir_teleop(brazo="derecho", usar_config=False)
    izq, q_izq, _ = construir_teleop(brazo="izquierdo", usar_config=False)
    compartir_obstaculos(der, q_der, izq, q_izq)
    assert der.ik.col.entorno.dinamicas and izq.ik.col.entorno.dinamicas
    assert all(c_.nombre.startswith("ARCTOS:") for c_ in der.ik.col.entorno.dinamicas)
    assert all(c_.nombre.startswith("FR5:") for c_ in izq.ik.col.entorno.dinamicas)
    # en reposo no se tocan
    assert der.ik.col.libre(q_der) and izq.ik.col.libre(q_izq)
    # un trozo del otro brazo justo en la mano del ARCTOS: colisión, y la IK lo empuja fuera
    m = izq.cadena.fk_marcos(q_izq)
    p = m[6][:3, :3] @ np.array([0.0, 0.15, 0.0]) + m[6][:3, 3]
    izq.ik.col.entorno.dinamicas = [Capsula("FR5:prueba", p, p + np.array([0.0, 0.0, 0.05]), 0.04)]
    assert any(b == "FR5:prueba" for _, b, _ in izq.ik.col.colisiones(q_izq))
    assert izq.ik._repulsion(q_izq, m) is not None
    izq.ik.col.entorno.dinamicas = None
    assert izq.ik.col.libre(q_izq)

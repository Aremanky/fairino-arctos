"""
Escenario del taladro — tecla [4].

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import numpy as np
import mujoco
import math

from config import *

from skill_shield.simulacion.objetos import OBJ_OCULTO
from skill_shield.simulacion.entorno_taller import TALLER_IMPRATIO, TALLER_NOSLIP_ITER
from skill_shield.brazo.ik import OFFSET_LOCAL_PALMA


# ══════════════════════════════════════════════════════════════════════════════
# ESCENARIO TALADRO — TECLA [4]
# ══════════════════════════════════════════════════════════════════════════════
# Un taladro de pie sobre una peana y un techo encima del robot. Está definido
# en escena_taladro.xml (lo GENERA taladro/preparar_taladro.py a partir de
# taladro/drill.obj) y se incluye desde arctos_mjcf.xml con:
#     <include file="escena_taladro.xml"/>
# Paso 1: aparecer / desaparecer con [4], poder coger el taladro y llevarlo al
# techo. (Paso 2: que la broca gire y deje el agujero al tocar el techo.)

class EntornoTaladro:
    """Muestra / esconde el techo, la peana y el taladro, y ayuda a sujetarlo."""

    def __init__(self, modelo, datos):
        self.disponible = False
        self.visible    = False
        self.sujeto     = False

        _id = lambda tipo, n: mujoco.mj_name2id(modelo, tipo, n)
        B, G, J = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_JOINT
        self.b_techo, self.b_soporte = _id(B, "taladro_techo"), _id(B, "taladro_soporte")
        self.b_taladro = _id(B, "taladro")
        self.j_libre   = _id(J, "taladro_libre")
        self.j_giro    = _id(J, "taladro_giro")
        self.g_bateria = _id(G, "taladro_col_bateria")
        if min(self.b_techo, self.b_soporte, self.b_taladro, self.j_libre, self.j_giro,
               self.g_bateria) < 0:
            print("⚠ Escenario taladro [4] NO disponible: ejecuta taladro/preparar_taladro.py "
                  "y añade <include file=\"escena_taladro.xml\"/> en arctos_mjcf.xml.")
            return

        self.m_techo   = modelo.body_mocapid[self.b_techo]
        self.m_soporte = modelo.body_mocapid[self.b_soporte]
        self.qa = modelo.jnt_qposadr[self.j_libre]
        self.da = modelo.jnt_dofadr[self.j_libre]
        self.da_giro = modelo.jnt_dofadr[self.j_giro]

        # Cuerpos del taladro (el propio + el rotor) y de la MANO (todo lo que
        # cuelga de mano_mount), para contar contactos mano ↔ taladro.
        def desciende_de(b, raiz):
            while b > 0:
                if b == raiz:
                    return True
                b = modelo.body_parentid[b]
            return False
        b_mano = _id(B, "mano_mount")
        self.cuerpos_taladro = {b for b in range(modelo.nbody) if desciende_de(b, self.b_taladro)}
        self.cuerpos_mano    = {b for b in range(modelo.nbody) if b_mano >= 0 and desciende_de(b, b_mano)}

        # Geoms que CHOCAN de techo, peana y taladro, por nombre (los visuales no).
        # Se les pone la misma "máscara" de colisión que usan la mano y el suelo:
        # si el modelo del robot usa otros bits (o un <default> con contype=0),
        # con contype=1 a secas el taladro atravesaría todo y caería al infinito.
        nombres_col = ("taladro_col_", "taladro_punta", "taladro_techo_geom", "taladro_soporte_geom")
        self.geoms_col = [g for g in range(modelo.ngeom)
                          if (mujoco.mj_id2name(modelo, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                          .startswith(nombres_col)]
        mascara = 0
        for g in range(modelo.ngeom):
            b = modelo.geom_bodyid[g]
            es_suelo = (b == 0 and modelo.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_PLANE))
            if b in self.cuerpos_mano or es_suelo:
                mascara |= int(modelo.geom_contype[g]) | int(modelo.geom_conaffinity[g])
        mascara = mascara or 1
        for g in self.geoms_col:
            modelo.geom_contype[g] = modelo.geom_conaffinity[g] = mascara
        self.colision_original = {g: (mascara, mascara) for g in self.geoms_col}
        print(f"🔩 Taladro: {len(self.geoms_col)} geoms de colisión (máscara {mascara}), "
              f"{len(self.cuerpos_mano)} cuerpos de mano")
        self.opt_original = (int(modelo.opt.cone), float(modelo.opt.impratio),
                             int(modelo.opt.noslip_iterations))
        # Altura de la base de la batería por debajo del origen del taladro
        self.alto_base = float(modelo.geom_size[self.g_bateria][2] - modelo.geom_pos[self.g_bateria][2])
        # Techo (su geom) y marcas de agujeros (mocap) — las usa la demo [5]
        self.g_techo = _id(G, "taladro_techo_geom")
        self.b_agujeros = [b for b in (_id(B, f"taladro_agujero_{k}") for k in range(32)) if b >= 0]
        self.m_agujeros = [int(modelo.body_mocapid[b]) for b in self.b_agujeros]

        self.disponible = True
        self.ocultar(modelo, datos)
        print("🔩 Escenario taladro cargado (pulsa [4] para mostrarlo).")

    # ── utilidades ────────────────────────────────────────────────────────────
    def _colisiones(self, modelo, activas: bool):
        for g, (ct, ca) in self.colision_original.items():
            modelo.geom_contype[g]     = ct if activas else 0
            modelo.geom_conaffinity[g] = ca if activas else 0

    @staticmethod
    def _altura_palma_vertical(modelo, datos):
        """Altura de la palma con el brazo totalmente vertical (para colocar el techo)."""
        ids = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}") for k in (1, 2, 3)]
        qa  = [modelo.jnt_qposadr[j] for j in ids]
        bm  = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, "mano_mount")
        guarda = [float(datos.qpos[i]) for i in qa]
        for i, v in zip(qa, (0.0, 0.0, 1.52)):          # joint3 = 1.52 → antebrazo recto
            datos.qpos[i] = v
        mujoco.mj_kinematics(modelo, datos)
        z = float((datos.xpos[bm] + datos.xmat[bm].reshape(3, 3) @ np.array(OFFSET_LOCAL_PALMA))[2])
        for i, v in zip(qa, guarda):
            datos.qpos[i] = v
        mujoco.mj_kinematics(modelo, datos)
        return z

    # ── mostrar / ocultar ─────────────────────────────────────────────────────
    def mostrar(self, modelo, datos):
        # Techo: por encima de la palma con el brazo vertical, lo justo para
        # llegar con la broca (o a la altura fija de config si se indica).
        z_techo = TALADRO_TECHO_Z if TALADRO_TECHO_Z is not None else \
            self._altura_palma_vertical(modelo, datos) + TALADRO_TECHO_HOLGURA
        datos.mocap_pos[self.m_techo]  = (TALADRO_TECHO_XY[0], TALADRO_TECHO_XY[1], z_techo + 0.015)
        datos.mocap_quat[self.m_techo] = [1.0, 0.0, 0.0, 0.0]

        # Peana y taladro de pie encima, con la broca mirando HACIA FUERA del robot
        # (así la empuñadura queda del lado del brazo).
        sop = np.array(TALADRO_POS_SOPORTE, dtype=float)
        self.recolocar_taladro(modelo, datos, forward=False)

        self._colisiones(modelo, True)
        modelo.opt.cone              = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        modelo.opt.impratio          = TALLER_IMPRATIO
        modelo.opt.noslip_iterations = TALLER_NOSLIP_ITER
        mujoco.mj_forward(modelo, datos)
        self.visible, self.sujeto = True, False
        print(f"🔩 Techo a z = {z_techo:.2f} m · taladro en la peana ({sop[0]:.2f}, {sop[1]:.2f}, {sop[2]:.2f})")

    def recolocar_taladro(self, modelo, datos, forward=True):
        """Peana en su sitio y taladro DE PIE encima, quieto (broca hacia fuera)."""
        sop = np.array(TALADRO_POS_SOPORTE, dtype=float)
        datos.mocap_pos[self.m_soporte]  = sop - [0.0, 0.0, 0.01]
        datos.mocap_quat[self.m_soporte] = [1.0, 0.0, 0.0, 0.0]
        yaw = math.atan2(sop[1], sop[0])
        datos.qpos[self.qa:self.qa + 3]     = sop + [0.0, 0.0, self.alto_base + 0.003]
        datos.qpos[self.qa + 3:self.qa + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        datos.qvel[self.da:self.da + 6]     = 0.0
        datos.qvel[self.da_giro]            = 0.0
        modelo.body_gravcomp[self.b_taladro] = 0.0
        if forward:
            mujoco.mj_forward(modelo, datos)

    def ocultar(self, modelo, datos):
        for k, m in enumerate(self.m_agujeros):          # los agujeros hechos por la demo [5]
            datos.mocap_pos[m] = (OBJ_OCULTO[0] + 3.0 + 0.1 * k, OBJ_OCULTO[1], OBJ_OCULTO[2])
        datos.mocap_pos[self.m_techo]   = OBJ_OCULTO
        datos.mocap_pos[self.m_soporte] = (OBJ_OCULTO[0] + 1.0, OBJ_OCULTO[1], OBJ_OCULTO[2])
        datos.qpos[self.qa:self.qa + 3]     = (OBJ_OCULTO[0] + 2.0, OBJ_OCULTO[1], OBJ_OCULTO[2])
        datos.qpos[self.qa + 3:self.qa + 7] = [1.0, 0.0, 0.0, 0.0]
        datos.qvel[self.da:self.da + 6]     = 0.0
        modelo.body_gravcomp[self.b_taladro] = 1.0       # flotando, quieto, sin caer
        self._colisiones(modelo, False)
        (modelo.opt.cone, modelo.opt.impratio, modelo.opt.noslip_iterations) = self.opt_original
        mujoco.mj_forward(modelo, datos)
        self.visible, self.sujeto = False, False

    def alternar(self, modelo, datos) -> bool:
        if self.visible:
            self.ocultar(modelo, datos)
        else:
            self.mostrar(modelo, datos)
        return self.visible

    # ── cada fotograma ────────────────────────────────────────────────────────
    def contactos_mano(self, modelo, datos) -> int:
        n = 0
        for i in range(datos.ncon):
            c = datos.contact[i]
            b1, b2 = modelo.geom_bodyid[c.geom1], modelo.geom_bodyid[c.geom2]
            if (b1 in self.cuerpos_mano and b2 in self.cuerpos_taladro) or \
               (b2 in self.cuerpos_mano and b1 in self.cuerpos_taladro):
                n += 1
        return n

    def actualizar(self, modelo, datos):
        """AGARRE ASISTIDO: mientras la mano lo toca con suficientes contactos,
        el taladro no pesa (gravcomp). Los dedos solo tienen que sujetarlo, no
        levantar 0.6 kg, que con la mano simulada acaba resbalando."""
        if not self.visible:
            return
        # Red de seguridad: si se ha caído muy abajo, vuelve a la peana
        if datos.qpos[self.qa + 2] < -0.5:
            print("⚠ El taladro se ha caído del escenario → vuelve a la peana")
            self.mostrar(modelo, datos)
            return
        if not TALADRO_ASISTENCIA_AGARRE:
            return
        sujeto = self.contactos_mano(modelo, datos) >= TALADRO_CONTACTOS_AGARRE
        if sujeto != self.sujeto:
            modelo.body_gravcomp[self.b_taladro] = 1.0 if sujeto else 0.0
            print("✊ Taladro SUJETO (agarre asistido)" if sujeto else "🖐️ Taladro SOLTADO")
            self.sujeto = sujeto

"""
Entorno de trabajo — tecla [3] (mesa con alicates, destornillador, bombilla y fusible).

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import numpy as np
import mujoco

from skill_shield.simulacion.objetos import OBJ_OCULTO


# ══════════════════════════════════════════════════════════════════════════════
# ENTORNO DE TRABAJO — TECLA [3]
# ══════════════════════════════════════════════════════════════════════════════
# Mini mesa de taller con alicates, destornillador, bombilla y fusible (más una
# bandeja de piezas, un portalámparas y un portafusibles fijos). Todo está
# definido en mesa_trabajo.xml, que se incluye desde arctos_mjcf.xml con:
#     <include file="mesa_trabajo.xml"/>
# Al arrancar queda escondido y sin colisiones; [3] lo coloca delante del brazo
# (con los objetos recolocados en su sitio) y [3] otra vez lo esconde.

# Centro de la SUPERFICIE de la mesa en el mundo (m): (x, y, z).
#   x → distancia delante del brazo (MÁS GRANDE = MÁS LEJOS del brazo)
#   y → desplazamiento lateral
#   z → altura de la superficie de trabajo
# Los objetos y la cámara se mueven solos con la mesa.
TALLER_POS_MESA = (-0.8, 0.0, 0.10)
# 27-oct: con el soporte de hombros la mesa baja a la altura de trabajo del brazo
# colgando; config.py la pone en TALLER_POS_MESA (solo si el modelo tiene ese soporte).
import config as _config
TALLER_POS_MESA = tuple(getattr(_config, "TALLER_POS_MESA", TALLER_POS_MESA))

TALLER_MESA = "taller_mesa"

# Pose de cada objeto RELATIVA al centro de la superficie de la mesa:
#   ((x, y, z) en metros, cuaternión de reposo (w, x, y, z))
# La z es la altura a la que se suelta (unos mm por encima de donde apoya).
TALLER_OBJETOS = {
    "taller_alicates":       ((-0.070,  0.100, 0.0085), (0.7071068, 0.0, 0.0, 0.7071068)),  # girados 90°
    "taller_destornillador": ((-0.030, -0.050, 0.0133), (0.9848078, 0.0, 0.0, 0.1736482)),  # girado 20°
    "taller_bombilla":       (( 0.062,  0.175, 0.0350), (1.0, 0.0, 0.0, 0.0)),              # en la bandeja
    "taller_fusible":        (( 0.060,  0.115, 0.0122), (1.0, 0.0, 0.0, 0.0)),              # en la bandeja
}

# Al mostrar el entorno, reencuadrar la cámara del visor para verlo entero
TALLER_ENCUADRAR_CAMARA = True

# Física de contacto "para agarrar" que se activa SOLO mientras la mesa está
# visible (al ocultarla se restauran los valores originales del modelo, así
# que el resto del comportamiento — pelota, destornillador [2], modo [M]... —
# no cambia):
#   · cono de fricción elíptico + impratio alto → el rozamiento se modela de
#     forma más realista y los objetos no "patinan" entre los dedos.
#   · noslip → elimina el deslizamiento lento de un objeto sujeto en pinza.
TALLER_FISICA_AGARRE = True
TALLER_IMPRATIO      = 10.0
TALLER_NOSLIP_ITER   = 3
TALLER_CAMARA = {
    # La cámara mira a un punto entre la base del brazo y la mesa; se calcula
    # a partir de TALLER_POS_MESA para que siga a la mesa si la mueves.
    "lookat":    (0.72 * TALLER_POS_MESA[0], TALLER_POS_MESA[1], TALLER_POS_MESA[2] + 0.08),
    "distancia": 0.95,
    "azimut":    225,
    "elevacion": -30,
}
TALLER_CAMARA = dict(getattr(_config, "TALLER_CAMARA", TALLER_CAMARA))   # 27-oct (soporte de hombros)


class EntornoTaller:
    """Muestra / esconde la mesa de trabajo y sus objetos en MuJoCo."""

    def __init__(self, modelo, datos):
        self.disponible = False
        self.visible    = False

        faltan = []
        mesa = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, TALLER_MESA)
        if mesa < 0 or modelo.body_mocapid[mesa] < 0:
            faltan.append(TALLER_MESA)

        # nombre → (body id, índice en qpos, índice en qvel) de su freejoint
        self.objetos = {}
        for nombre in TALLER_OBJETOS:
            bid = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, nombre)
            if bid < 0 or modelo.body_jntnum[bid] != 1 or \
               int(modelo.jnt_type[modelo.body_jntadr[bid]]) != int(mujoco.mjtJoint.mjJNT_FREE):
                faltan.append(nombre)
                continue
            j = modelo.body_jntadr[bid]
            self.objetos[nombre] = (bid, modelo.jnt_qposadr[j], modelo.jnt_dofadr[j])

        if faltan:
            print(f"⚠ Entorno de trabajo [3] NO disponible, faltan en el modelo: {faltan}. "
                  f"¿Está <include file=\"mesa_trabajo.xml\"/> en arctos_mjcf.xml?")
            return

        self.mocap_id = modelo.body_mocapid[mesa]

        # Geoms que chocan (group 0) de la mesa y los objetos, con sus valores
        # originales del XML, para poder apagar/encender sus colisiones.
        cuerpos = {mesa} | {bid for bid, _, _ in self.objetos.values()}
        self.colision_original = {
            g: (int(modelo.geom_contype[g]), int(modelo.geom_conaffinity[g]))
            for g in range(modelo.ngeom)
            if modelo.geom_bodyid[g] in cuerpos and modelo.geom_group[g] == 0
        }
        self.gravcomp_original = {bid: float(modelo.body_gravcomp[bid])
                                  for bid, _, _ in self.objetos.values()}

        # Opciones de física originales, para restaurarlas al ocultar la mesa
        self.opt_original = (int(modelo.opt.cone), float(modelo.opt.impratio),
                             int(modelo.opt.noslip_iterations))

        self.disponible = True
        self.ocultar(modelo, datos)
        print("🧰 Entorno de trabajo cargado (pulsa [3] para mostrarlo).")

    def _colisiones(self, modelo, activas: bool):
        for g, (ct, ca) in self.colision_original.items():
            modelo.geom_contype[g]     = ct if activas else 0
            modelo.geom_conaffinity[g] = ca if activas else 0

    def mostrar(self, modelo, datos):
        base = np.array(TALLER_POS_MESA, dtype=float)
        datos.mocap_pos[self.mocap_id]  = base
        datos.mocap_quat[self.mocap_id] = [1.0, 0.0, 0.0, 0.0]
        for nombre, (bid, qa, da) in self.objetos.items():
            rel, quat = TALLER_OBJETOS[nombre]
            q = np.array(quat, dtype=float)
            datos.qpos[qa:qa + 3]     = base + np.array(rel, dtype=float)
            datos.qpos[qa + 3:qa + 7] = q / np.linalg.norm(q)
            datos.qvel[da:da + 6]     = 0.0
            modelo.body_gravcomp[bid] = self.gravcomp_original[bid]
        self._colisiones(modelo, True)
        if TALLER_FISICA_AGARRE:
            modelo.opt.cone              = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
            modelo.opt.impratio          = TALLER_IMPRATIO
            modelo.opt.noslip_iterations = TALLER_NOSLIP_ITER
        mujoco.mj_forward(modelo, datos)
        self.visible = True

    def ocultar(self, modelo, datos):
        # Escondido bajo el suelo, sin colisiones y "flotando" (gravcomp) para
        # que no se quede cayendo indefinidamente mientras no se usa.
        datos.mocap_pos[self.mocap_id] = OBJ_OCULTO
        self._colisiones(modelo, False)
        (modelo.opt.cone, modelo.opt.impratio,
         modelo.opt.noslip_iterations) = self.opt_original
        for k, (bid, qa, da) in enumerate(self.objetos.values()):
            datos.qpos[qa:qa + 3]     = (OBJ_OCULTO[0] + 0.5 * (k + 1), OBJ_OCULTO[1], OBJ_OCULTO[2])
            datos.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
            datos.qvel[da:da + 6]     = 0.0
            modelo.body_gravcomp[bid] = 1.0
        mujoco.mj_forward(modelo, datos)
        self.visible = False

    def alternar(self, modelo, datos) -> bool:
        if self.visible:
            self.ocultar(modelo, datos)
        else:
            self.mostrar(modelo, datos)
        return self.visible

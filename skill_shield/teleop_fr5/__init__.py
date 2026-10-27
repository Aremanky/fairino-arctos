"""
Teleoperación del FAIRINO FR5 con dos OAK-D (muñeca + codo) — versión 2 (27-oct).

Módulos (todos en numpy puro, sin MuJoCo, para poder probarlos sin la sala):

  cinematica          cadena cinemática del FR5 leída del MJCF: FK, jacobiana, parámetros DH
                      equivalentes e IK ANALÍTICA (las 8 configuraciones de un brazo tipo UR)
  fusion_muneca_codo  hombro/codo/muñeca 3D con las dos cámaras: pesos por oclusión de la
                      palma y por escorzo, triangulación por rayos, filtro de Kalman
  retargeting         medida humana (con límites anatómicos reales) → objetivo del robot
  ik_teleop           elección de configuración, límites, colisiones y "otro camino"
  teleop_brazo        orquestador por fotograma (lo que llama Skill_Shield_Nivel_6.py)
  simulador           brazo humano sintético + cámaras sintéticas para probar todo sin hardware
"""

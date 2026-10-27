# Decoración corporativa Elecnor

Va en `arctos_description-main/arctos_urdf_description/urdf/decoracion/`, junto a `arctos_mjcf.xml`.

| Fichero | Dónde se incluye | Qué añade |
|---|---|---|
| `deco_assets.xml` | `arctos_mjcf.xml` (raíz) | Materiales y texturas `deco_*` |
| `deco_escena_base.xml` | `arctos_mjcf.xml` (raíz) | Cabecera con rótulo y cenefa en la pared del fondo, pared lateral derecha con logo, cinta de seguridad en el suelo, placa con logo delante de la peana, logo I+D+i en la peana del brazo, luz de relleno |
| `deco_mesa.xml` | dentro del body `taller_mesa` (`mesa_trabajo.xml`) | Canto naranja, faldón azul con logos, panel trasero de taller con rótulo, logos y herramienta colgada |
| `deco_taladro_peana.xml` | dentro del body `taladro_soporte` (`escena_taladro.xml`) | Columna corporativa bajo la placa del taladro, con logos verticales y base en el suelo |
| `deco_taladro_techo.xml` | dentro del body `taladro_techo` (`escena_taladro.xml`) | Faldón azul/naranja con logos en las 4 caras y marco de cinta de seguridad alrededor de la zona de taladrado |
| `soporte_hombros.xml` | `fairino_mjcf.xml` (raíz) | Soporte del FR5 (27-oct): bastidor con ruedas, columna, cabezal en T y dos hombros a 45°; el FR5 va en el hombro DERECHO. Lo genera `crear_soporte_hombros.py` (medidas, colores y pose de `base_link`) |
| `texturas/*.png` | — | Rótulos, cinta de seguridad y panel perforado (generados) |

## Reglas que cumple todo (para no tocar la funcionalidad)

- Solo geoms: ningún `<body>`, `<joint>` ni mocap nuevo → mismos `qpos`, ids de cuerpos y mocap.
- `contype="0" conaffinity="0"`, `group="2"` y `density="0"` → no chocan ni pesan.
  `EntornoTaller` solo toca los geoms del grupo 0 y `EntornoTaladro` los que empiezan por
  `taladro_col_`, `taladro_punta`, `taladro_techo_geom` y `taladro_soporte_geom`: estos no.
- Brazo, peana, taladro, mesa y techo siguen exactamente donde estaban: la demo [5] no cambia.

## Si cambias algo de config.py

- `TALADRO_POS_SOPORTE` (altura): mueve la base de suelo de `deco_taladro_peana.xml` (marca [Z]).
- `TALADRO_DEMO_AGUJERO_XY` o `TALADRO_TECHO_XY`: mueve el marco de `deco_taladro_techo.xml` (marca [A]).
- Altura del suelo: las z marcadas con [H] en `fairino_mjcf.xml` y `deco_escena_base.xml`, la base
  [Z] de `deco_taladro_peana.xml` y `SOPORTE_SUELO_Z` sale sola de `fairino_mjcf.xml` (geom `floor`).
  Con el soporte de hombros el suelo está en z = -0.975 (antes -0.70 con la peana).

## Si regeneras `escena_taladro.xml` con `taladro/preparar_taladro.py`

Vuelve a poner sus dos líneas `<include file="decoracion/..."/>` (una dentro de `taladro_techo`
y otra dentro de `taladro_soporte`, después de su geom).

## Logos

Los logos se dibujan en proporción 3:1, igual que los que ya había en la pared. Si alguno
sale estirado, cambia la relación de su `size` (ancho/alto) en el `<geom>` correspondiente.

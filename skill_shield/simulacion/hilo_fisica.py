"""
Física en tiempo real en un hilo propio (idea tomada del proyecto TRON 2).

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import mujoco
import time
import threading
from contextlib import contextmanager

from config import *


# ══════════════════════════════════════════════════════════════════════════════
# FÍSICA EN TIEMPO REAL — hilo propio (idea tomada del proyecto TRON 2)
# ══════════════════════════════════════════════════════════════════════════════

class HiloFisica(threading.Thread):
    """
    Avanza la simulación MuJoCo al ritmo del RELOJ REAL y refresca el visor a
    FISICA_HZ, en un hilo aparte. Antes se hacían 4 mj_step por fotograma de
    cámara: a ~8 FPS la simulación iba al ~8 % del tiempo real (robot "a cámara
    lenta" y REAL siempre por detrás de CMD).

    Reparto del acceso a `datos` (un único candado):
      · El bucle principal TIENE el candado casi siempre (lee y escribe datos:
        ctrl, IK, pinzas, teclas...), y lo SUELTA solo durante lo lento (esperar
        a la cámara, MediaPipe) con `with fisica.soltar():`.
      · Mientras está suelto, este hilo avanza la física lo que haga falta para
        alcanzar al reloj real y refresca el visor.
    Así nunca se tocan `datos` desde dos hilos a la vez y no hace falta reordenar
    el bucle principal.
    """

    def __init__(self, modelo, datos, visor, antes_de_paso=None):
        super().__init__(daemon=True)
        self.modelo, self.datos, self.visor = modelo, datos, visor
        self.antes_de_paso = antes_de_paso   # PASO 7.1: se llama antes de CADA mj_step
        self.lock   = threading.Lock()
        self.activo = True
        self.factor = 0.0          # tiempo simulado / tiempo real (1.00 = tiempo real)
        self.espera_candado = 0.0  # FLUIDEZ: s que el bucle principal esperó a recuperar el candado
        # 28-oct (latencia): el bucle principal esperaba 40-49 ms por vuelta a que este hilo
        # terminara su lote de mj_step + visor.sync() con el candado cogido. Ahora, en cuanto el
        # bucle principal pide el candado, este hilo corta el lote tras el mj_step en curso (lo
        # que falte se simula en la siguiente vuelta) y no hace el sync de esa vuelta.
        self._quiere_principal = False

    def empezar(self):
        """Lo llama el bucle principal: se queda el candado y arranca el hilo."""
        self.lock.acquire()
        self.start()

    def parar(self):
        """Lo llama el bucle principal al salir del modo teleoperación."""
        self.activo = False
        self.lock.release()
        self.join(timeout=1.0)

    @contextmanager
    def soltar(self):
        """El bucle principal suelta la simulación mientras hace algo lento."""
        self.lock.release()
        try:
            yield
        finally:
            _t_esp = time.perf_counter()
            self._quiere_principal = True
            self.lock.acquire()
            self._quiere_principal = False
            # FLUIDEZ: si la física estaba a mitad de un lote de mj_step, el bucle de la
            # cámara se queda aquí esperando; lo mide PerfilBucle.
            self.espera_candado += time.perf_counter() - _t_esp

    def run(self):
        ts      = float(self.modelo.opt.timestep)
        periodo = 1.0 / FISICA_HZ
        deuda   = 0.0                               # tiempo real aún no simulado (s)
        t_ant   = time.perf_counter()
        t_med, sim_med = t_ant, 0.0
        while self.activo:
            t0 = time.perf_counter()
            with self.lock:
                if not self.activo:
                    break
                ahora = time.perf_counter()
                # Si el bucle principal retuvo el candado mucho rato, no se intenta
                # recuperar TODO de golpe (se tiraría demasiado tiempo calculando).
                deuda = min(deuda + (ahora - t_ant), FISICA_RETRASO_MAX)
                t_ant = ahora
                n = int(deuda / ts)
                hechos = 0
                for _ in range(n):
                    if self._quiere_principal:             # 28-oct: el bucle principal espera
                        break
                    if self.antes_de_paso is not None:     # PASO 7.1: comando continuo
                        self.antes_de_paso(self.modelo, self.datos, ts)
                    mujoco.mj_step(self.modelo, self.datos)
                    hechos += 1
                deuda   -= hechos * ts
                sim_med += hechos * ts
                if not self._quiere_principal and self.visor.is_running():
                    self.visor.sync()
            if ahora - t_med >= 1.0:
                self.factor = sim_med / (ahora - t_med)
                t_med, sim_med = ahora, 0.0
            espera = periodo - (time.perf_counter() - t0)
            if espera > 0:
                time.sleep(espera)

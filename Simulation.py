import json
import os
import random

from bonsai_common import SimulatorSession, Schema
import dotenv
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

from Chuveiro_Turbinado import MalhaFechada, MalhaAberta, ChuveiroTurbinado, PID_p2

class ChuveiroTurbinadoSimulation(SimulatorSession):
    
    def reset(
        self,
        Sr_0: float = 50,
        Sa_0: float = 50,
        xq_0: float = 0.3,
        xf_0: float = 0.5,
        xs_0: float = 0.4672,
        Fd_0: float = 0,
        Td_0: float = 25,
        Tinf: float = 25,
        T0: list = [50,  30 ,  30,  30],
        SPh_0: float = 60,
        SPT4a_0: float = 37
    ):
        """
        Chuveiro turbinado para simulação.
        ---------------------------------
        Parâmetros:
        Sr_0: seletor que define a fração da resistência elétrica utilizada no aquecimento do tanque de mistura.
        Sa_0: seletor que define a fração para aquecimento à gás do boiler.
        xq_0: abertura da válvula para entrada da corrente quente aquecida pelo boiler.
        xf_0: abertura da válvula para entrada da corrente fria.
        xs_0: abertura da válvula para saída da corrente na tubulação final.
        Fd_0: vazão da corrente de distúrbio.
        Td_0: temperatura da corrente de distúrbio.
        Tinf: temperatura ambiente.
        T0: condições iniciais da simulação (nível do tanque, temperatura do tanque de mistura, 
            temperatura de aquecimento do boiler, temperatura de saída.
        SPh_0: setpoint inicial do nível do tanque de mistura da corrente fria com quente.
        SPT4a_0: setpoint inicial da temperatura de saída do sistema.
        """
        
        self.Sr = Sr_0
        self.Sa = Sa_0
        self.xq = xq_0
        self.xf = xf_0
        self.xs = xs_0
        self.Fd = Fd_0
        self.Td = Td_0
        self.Tinf = Tinf
        self.T0 = T0
        self.SPh = SPh_0
        self.SPT4a = SPT4a_0
        
        # Preciso adaptar esta parte:
        # Quanto tempo vai durar um episódio? Quais variáveis serão constantes ou não?
        TU = np.array(
        [    #Time,  SP(T4a),   Sa,    xq,  SP(h),      Xs,   Fd,  Td,  Tinf
              [  0,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
              [ 10,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
              [ 20,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
              [ 30,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
              [ 40,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
              [ 50,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
              [ 60,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
              [ 70,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
              [ 80,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
              [ 90,       38,   50,   0.3,     70,   0.4672,   1,  28,   20],
              [100,       38,   50,   0.3,     70,   0.4672,   1,  28,   20],
        ])   

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, TU, Kp_T4a = [20.63, 1], Ti_T4a = [1.53, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = 0.01)
        self.TT, self.YY, self.UU = malha_fechada.solve_system()
        
        # Valores finais das variáveis de estado (no tempo final):
        self.h = self.YY[:,0]
        self.T4a = self.YY[:,-1]
        self.T3 = self.YY[:,1]
        
        # Valores finais das variáveis manipuladas e distúrbios:
        self.Sr = self.UU[:,0]
        self.Sa = self.UU[:,1]
        self.xq = self.UU[:,2]
        self.xf = self.UU[:,3]
        self.xs = self.UU[:,4]
        self.Fd = self.UU[:,5]
        self.Td = self.UU[:,6]
        self.Tinf = self.UU[:,7]
    
        # Falta calcular o custo do banho (levar em conta apenas energia elétrica ou o gás do boiler também?)
        # Cálculo do índice de qualidade do banho:
        self.T4a_final = self.YY[:,-1][-1]
        self.xs_final = self.UU[:,4][-1]
        self.Fs_final = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))
        self.iqb = malha_fechada.compute_iqb(self.T4a_final, self.Fs_final)

    def episode_start(self, config: Schema) -> None:
        
        # Esse config é um arquivo json?
        self.reset(
            Sr_0 = config.get('initial_electrical_resistence_fraction'),
            Sa_0 = config.get('initial_gas_boiler_fraction'),
            xq_0 = config.get('initial_hot_valve_opening'),
            xf_0 = config.get('initial_cold_valve_opening'),
            xs_0 = config.get('initial_out_valve_opening'),
            Fd_0 = config.get('initial_disturbance_current_flow'),
            Td_0 = config.get('initial_disturbance_temperature'),
            Tinf = config.get('initial_room_temperature'),
            T0 = config.get('initial_conditions'),
            SPh_0 = config.get('initial_setpoint_tank_level'),
            SPT4a_0 = config.get('initial_setpoint_final_temperature'),
        )
        
    def step(self):
        
        # Preciso adaptar esta parte:
        TU = np.array(
        [    #Time,  SP(T4a),   Sa,    xq,  SP(h),      Xs,   Fd,  Td,  Tinf
              [  0,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
              [ 10,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
              [ 20,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
              [ 30,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
              [ 40,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
              [ 50,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
              [ 60,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
              [ 70,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
              [ 80,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
              [ 90,       38,   50,   0.3,     70,   0.4672,   1,  28,   20],
              [100,       38,   50,   0.3,     70,   0.4672,   1,  28,   20],
        ]) 

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, TU, Kp_T4a = [20.63, 1], Ti_T4a = [1.53, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = 0.01)
        self.TT, self.YY, self.UU = malha_fechada.solve_system()
        
        # Valores finais das variáveis de estado (no tempo final):
        self.h = self.YY[:,0]
        self.T4a = self.YY[:,-1]
        self.T3 = self.YY[:,1]
        
        # Valores finais das variáveis manipuladas e distúrbios:
        self.Sr = self.UU[:,0]
        self.Sa = self.UU[:,1]
        self.xq = self.UU[:,2]
        self.xf = self.UU[:,3]
        self.xs = self.UU[:,4]
        self.Fd = self.UU[:,5]
        self.Td = self.UU[:,6]
        self.Tinf = self.UU[:,7]
    
        # Falta calcular o custo do banho (levar em conta apenas energia elétrica ou o gás do boiler também?)
        # Cálculo do índice de qualidade do banho:
        self.T4a_final = self.YY[:,-1][-1]
        self.xs_final = self.UU[:,4][-1]
        self.Fs_final = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))
        self.iqb = malha_fechada.compute_iqb(self.T4a_final, self.Fs_final)
        
    def episode_step(self, action: Schema) -> None:
        
        self.SPh = action.get('setpoint_tank_level')
        self.SPT4a = action.get('setpoint_final_temperature')

        self.step()

    def get_state(self):
        
        return {  
            'electrical_resistence_fraction': self.Sr,
            'gas_boiler_fraction': self.Sa,
            'hot_valve_opening': self.xq,
            'cold_valve_opening': self.xf,
            'out_valve_opening': self.xs,
            'disturbance_current_flow': self.Fd,
            'disturbance_temperature': self.Td,
            'room_temperature': self.Tinf,
            'setpoint_tank_level': self.SPh,
            'setpoint_final_temperature': self.SPT4a
        }
    
    def halted(self) -> bool:
        return False

    def get_interface(self) -> SimulatorInterface:
        """Register sim interface."""

        with open('interface.json', 'r') as infile:
            interface = json.load(infile)

        return SimulatorInterface(
            name=interface['name'],
            timeout=interface['timeout'],
            simulator_context=self.get_simulator_context(),
            description=interface['description'],
        )


def main():

    workspace = os.getenv('SIM_WORKSPACE')
    access_key = os.getenv('SIM_ACCESS_KEY')

    # values in `.env`, if they exist, take priority over environment variables
    dotenv.load_dotenv('.env', override=True)

    if workspace is None:
        raise ValueError('The Bonsai workspace ID is not set.')
    if access_key is None:
        raise ValueError('The access key for the Bonsai workspace is not set.')

    config = BonsaiClientConfig(workspace=workspace, access_key=access_key)

    chuveiro_sim = ChuveiroTurbinadoSimulation(config)

    chuveiro_sim.reset()

    while chuveiro_sim.run():
        continue


if __name__ == "__main__":
    main()
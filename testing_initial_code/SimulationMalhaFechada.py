import json
import os
import random
import numpy as np
import copy
import time as tm
import matplotlib.pyplot as plt

#from bonsai_common import SimulatorSession, Schema
#import dotenv
#from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
#from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

from Chuveiro_Turbinado import MalhaFechada, ChuveiroTurbinado, PID_p2

# class ChuveiroTurbinadoSimulation(SimulatorSession):
class ChuveiroTurbinadoSimulation():
    
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
        
        self.time = 2
        self.dt = 0.01
        self.time_sample = 2 #minutos
        
        # Definindo TU:
        # Time, SP(T4a), Sa, xq, SP(h), xs, Fd, Td, Tinf
        TU = np.array(
        [   
              [0, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf],
              [self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]
        ])

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, TU, Kp_T4a = [20.63, 1], Ti_T4a = [1.53, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = self.dt)
        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas
        self.TT, self.YY, self.UU = malha_fechada.solve_system()
        
        # Valores finais das variáveis de estado (no tempo final):
        self.h = self.YY[:,0][-1]
        self.T4a = self.YY[:,-1][-1]
        self.T3 = self.YY[:,1][-1]
        self.Tq = self.YY[:,2][-1]
        
        # Valores finais das variáveis manipuladas e distúrbios:
        self.Sr = self.UU[:,0][-1]
        self.Sa = self.UU[:,1][-1]
        self.xq = self.UU[:,2][-1]
        self.xf = self.UU[:,3][-1]
        self.xs = self.UU[:,4][-1]
        self.Fd = self.UU[:,5][-1]
        self.Td = self.UU[:,6][-1]
        self.Tinf = self.UU[:,7][-1]
    
        # Cálculo do índice de qualidade do banho:
        self.iqb = malha_fechada.compute_iqb(self.YY[:,-1], # T4a
                                             self.UU[:,4], # xs
                                             self.TT) or 0
        if np.isnan(self.iqb) or self.iqb == None or np.isinf(abs(self.iqb)):
            self.iqb = 0

        # Cálculo do custo do banho:
        self.custo_eletrico, self.custo_gas = malha_fechada.custo_banho(self.UU[:,0], # Sr
                                                                        self.UU[:,2], # xq
                                                                        self.UU[:,3], # xf
                                                                        self.YY[:,2], # Tq
                                                                        self.UU[:,7], # Tinf
                                                                        self.TT,
                                                                        self.dt)

        # Cálculo do custo da água:
        self.custo_agua = malha_fechada.custo_agua(self.UU[:,4], #xs
                                                   self.TT,
                                                   self.dt)

        # Vazão final Fs:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))

        self.TU = TU
        self.TU_list = copy.deepcopy(TU)

        # Salvar o estado atual:
        self.last_TU = copy.deepcopy(np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])

    # def episode_start(self, config: Schema) -> None:
    def episode_start(self, config) -> None:
        
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
            SPT4a_0 = config.get('initial_setpoint_final_temperature')
        )
        
    def step(self):
        
        self.time += self.time_sample 
        
        # if self.time >= self.time_sample *2 + 1: #2
        #     self.TU = np.append(self.last_TU, np.array([[self.time_sample, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        #     self.T0 = self.last_T0 # h, T3, Tq, T4a
        #     print(self.TU)
        #     print(self.time)
        # else:
        #     self.TU = np.append(self.TU, np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        #     print(self.TU)

        # self.TU_list = np.append(self.TU_list, np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        # print("TU List: ", self.TU_list)

        self.TU = np.append(self.last_TU, np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        self.T0 = self.last_T0 # h, T3, Tq, T4a
        self.TU_list = np.append(self.TU_list, np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        print(self.TU)
        print(self.T0)
        print(self.time)
        print(self.TU_list)

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, self.TU, Kp_T4a = [20.63, 1], Ti_T4a = [1.53, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = self.dt)
        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas
        self.TT, self.YY, self.UU = malha_fechada.solve_system()

        # Valores finais das variáveis de estado (no tempo final):
        self.h = self.YY[:,0][-1]
        self.T4a = self.YY[:,-1][-1]
        self.T3 = self.YY[:,1][-1]
        self.Tq = self.YY[:,2][-1]
        
        # Valores finais das variáveis manipuladas e distúrbios:
        self.Sr = self.UU[:,0][-1]
        self.Sa = self.UU[:,1][-1]
        self.xq = self.UU[:,2][-1]
        self.xf = self.UU[:,3][-1]
        self.xs = self.UU[:,4][-1]
        self.Fd = self.UU[:,5][-1]
        self.Td = self.UU[:,6][-1]
        self.Tinf = self.UU[:,7][-1]
    
        # Cálculo do índice de qualidade do banho:
        self.iqb = malha_fechada.compute_iqb(self.YY[:,-1], # T4a
                                             self.UU[:,4], # xs
                                             self.TT) or 0
        if np.isnan(self.iqb) or self.iqb == None or np.isinf(abs(self.iqb)):
            self.iqb = 0

        # Cálculo do custo do banho:
        self.custo_eletrico, self.custo_gas = malha_fechada.custo_banho(self.UU[:,0], # Sr
                                                                        self.UU[:,2], # xq
                                                                        self.UU[:,3], # xf
                                                                        self.YY[:,2], # Tq
                                                                        self.UU[:,7], # Tinf
                                                                        self.TT,
                                                                        self.dt)

        # Cálculo do custo da água:
        self.custo_agua = malha_fechada.custo_agua(self.UU[:,4], #xs
                                                   self.TT,
                                                   self.dt)

        # Vazão final Fs:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))
                
        # Salvar o estado atual:
        self.last_TU = copy.deepcopy(np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])

    # def episode_step(self, action: Schema) -> None:
    def episode_step(self, action) -> None:
        
        self.xq = action.get('hot_valve_opening')
        self.xs = action.get('out_valve_opening')
        self.SPT4a = action.get('setpoint_final_temperature')
        self.Sa = action.get('gas_boiler_fraction')
        self.Sph = action.get('setpoint_tank_level')
        self.Fd = action.get('disturbance_current_flow')
        self.Td = action.get('disturbance_temperature')
        self.Tinf = action.get('room_temperature')

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
            'tank_level': self.h,
            'setpoint_final_temperature': self.SPT4a,
            'final_temperature': self.T4a,
            'flow_out': self.Fs,
            'final_boiler_temperature': self.Tq,
            'final_temperature_tank': self.T3,
            'quality_of_shower': self.iqb,
            'electrical_cost_shower': self.custo_eletrico,
            'gas_cost_shower': self.custo_gas,
            'cost_water': self.custo_agua,
        }
    
    def halted(self) -> bool:
        
        if self.T4a > 63:
            return True
        else:
            return False

    """
    def get_interface(self) -> SimulatorInterface:
        # Register sim interface.

        with open('interface.json', 'r') as infile:
            interface = json.load(infile)

        return SimulatorInterface(
            name=interface['name'],
            timeout=interface['timeout'],
            simulator_context=self.get_simulator_context(),
            description=interface['description'],
        )
    """

def main():

    workspace = os.getenv('SIM_WORKSPACE')
    access_key = os.getenv('SIM_ACCESS_KEY')

    # values in `.env`, if they exist, take priority over environment variables
    # dotenv.load_dotenv('.env', override=True)

    # if workspace is None:
    #      raise ValueError('The Bonsai workspace ID is not set.')
    # if access_key is None:
    #     raise ValueError('The access key for the Bonsai workspace is not set.')

    # config = BonsaiClientConfig(workspace=workspace, access_key=access_key)
    config = None

    chuveiro_sim = ChuveiroTurbinadoSimulation(config)

    chuveiro_sim.reset()

    while chuveiro_sim.run():
        continue
        
def main_test():
    
    chuveiro_sim = ChuveiroTurbinadoSimulation()
    chuveiro_sim.reset()
    state = chuveiro_sim.get_state()

    q_list = []
    T4_list = []
    
       #Time,  SP(T4a),   Sa,    xq,  SP(h),      Xs,   Fd,  Td,  Tinf
    TU=[[ 4,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
        [ 6,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [ 8,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [10,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
        [12,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
        [14,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
        [16,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
        [18,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
        [20,       38,   50,   0.3,     70,   0.4672,   1,  28,   20],
        [22,       38,   50,   0.3,     70,   0.4672,   1,  28,   20]]   
    
    for i in range(0, 10):
        
        if chuveiro_sim.halted():
            break
            
        action = {
            'hot_valve_opening': TU[i][3],
            'out_valve_opening': TU[i][-4],
            'setpoint_final_temperature': TU[i][1],
            'gas_boiler_fraction': TU[i][2],
            'setpoint_tank_level': TU[i][4],
            'disturbance_current_flow': TU[i][6],
            'disturbance_temperature': TU[i][7],
            'room_temperature': TU[i][8],
        }
            
        chuveiro_sim.episode_step(action)
        state = chuveiro_sim.get_state()
        print('')
        print(state)
        q_list.append(state['quality_of_shower'])
        T4_list.append(state['final_temperature'])

    plt.figure(figsize=(10,7))
    plt.subplot(2,1,1)
    plt.plot([i for i in range(len(q_list))], q_list)
    plt.subplot(2,1,2)
    plt.plot([i for i in range(len(T4_list))], T4_list)
    plt.show()
    
if __name__ == "__main__":
    main_test()
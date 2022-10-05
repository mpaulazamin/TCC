import json
import os
import random
import numpy as np

#from bonsai_common import SimulatorSession, Schema
#import dotenv
#from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
#from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

from Chuveiro_Turbinado import MalhaFechada, MalhaAberta, ChuveiroTurbinado, PID_p2

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
        
        self.time = 1
        
        # Definindo TU:
        # Time,  SP(T4a),   Sa,    xq,  SP(h),      Xs,   Fd,  Td,  Tinf
        TU = np.array(
        [   
              [0, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf],
              [self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]
        ])

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, TU, Kp_T4a = [20.63, 1], Ti_T4a = [1.53, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = 0.01)
        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas
        self.TT, self.YY, self.UU = malha_fechada.solve_system()
        
        # Valores finais das variáveis de estado (no tempo final):
        self.h = self.YY[:,0][-1]
        self.T4a = self.YY[:,-1][-1]
        self.T3 = self.YY[:,1][-1]
        
        # Valores finais das variáveis manipuladas e distúrbios:
        self.Sr = self.UU[:,0][-1]
        self.Sa = self.UU[:,1][-1]
        self.xq = self.UU[:,2][-1]
        self.xf = self.UU[:,3][-1]
        self.xs = self.UU[:,4][-1]
        self.Fd = self.UU[:,5][-1]
        self.Td = self.UU[:,6][-1]
        self.Tinf = self.UU[:,7][-1]
    
        # Falta calcular o custo do banho (levar em conta apenas energia elétrica ou o gás do boiler também?)
        # Cálculo do índice de qualidade do banho:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))
        self.iqb = malha_fechada.compute_iqb(self.T4a, self.Fs)

        self.TU = TU

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
            SPT4a_0 = config.get('initial_setpoint_final_temperature'),
        )
        
    def step(self):
        
        self.time += 10
        
        self.TU = np.append(self.TU, np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, self.TU, Kp_T4a = [20.63, 1], Ti_T4a = [1.53, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = 0.01)
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
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))
        self.iqb = malha_fechada.compute_iqb(self.T4a, self.Fs)

        # Custo do banho:
        # Custo elétrico considerando 10 min de banho e utilizando o Sr final obtido a cada step: 
        self.potencia_eletrica = 5.5 # potência média de um chuveiro é 5500 W = 5.5 KW
        self.custo_kwh = 0.142
        self.tempo_banho = 10 / 60
        self.custo_eletrico = self.potencia_eletrica * (self.Sr / 100) * self.custo_kwh * self.tempo_banho

        # Custo do gás considerando 10 min de banho, o volume de água que está no boiler, e a temperatura T3 do boiler:
        # self.volume_boiler = 0.5 * self.h
        # vazão é em L/min, 1L de água equivale a aproximadamente 1000g
        # calcular quanto gás é necessário para aquecer a vazão (quantidade de litros que entra no tanque por minuto)
        t1 = self.xq ** 0.2
        t2 = t1 ** 2
        t5 = self.xf ** 2 
        t7 = self.xq ** 2 
        t9 = t1 * self.xq * t7
        t10 = t9 * t5
        t14 = t7 ** 2 
        t16 = t2 * t7 * t14 
        t17 = t16 * t5 
        t20 = np.sqrt(0.9e1 * t10 + 0.50e2 * t17) 
        t26 = t5 ** 2
        t38 = (np.sqrt(-0.1e1 * (-0.180e3 -0.45e2 * t5 -0.250e3 * t10 - 0.1180e4 * t9 + 0.60e2 * t20) / (0.6552e4 * t10 + 0.648e3 
               * t5 + 0.16400e5* t17 + 0.900e3 * t9 * t26 + 0.2500e4 * t16 * t26 + 0.81e2 * t26 + 0.55696e5 * t16 + 0.16992e5 
               * t9 + 0.1296e4)))
        self.Fq = 60 * t1 * t2 * self.xq * t38

        print(self.xq, self.Fq)

        self.custo_botijao_kg = 49.19 / 13 # em reais/kg, considerando botijão de 13kg
        self.calor_combustao_gas = 6000 # (kcal/kg)
        self.c_agua = 1 # (cal/g)
        self.Q = self.Fq * 1000 * self.c_agua * (self.Tq - self.Tinf) # volume em L equivale a kg então multiplica-se por 1000 para corrigir a unidade
        self.quantidade_gas = (self.Q / 1000) / self.calor_combustao_gas
        self.custo_gas_por_min = self.custo_botijao_kg * self.quantidade_gas
        self.custo_gas = self.custo_gas_por_min * 10 # custo total de gás para banho de 10 min

        print(self.custo_eletrico, self.custo_gas)
        
    # def episode_step(self, action: Schema) -> None:
    def episode_step(self, action) -> None:
        
        self.xq = action.get('hot_valve_opening')
        # self.xf = action.get('cold_valve_opening')
        self.xs = action.get('out_valve_opening')
        self.SPT4a = action.get('setpoint_final_temperature')
        self.Sa = action.get('gas_boiler_fraction')

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
            'setpoint_final_temperature': self.SPT4a,
            'quality_of_shower': self.iqb,
            'final_temperature': self.T4a 
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
    
    #Time,  SP(T4a),   Sa,    xq,  SP(h),      Xs,   Fd,  Td,  Tinf
    TU=[[ 10,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
        [ 20,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [ 30,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [ 40,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
        [ 50,       38,   50,   0.3,     70,   0.4672,   1,  25,   25],
        [ 60,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
        [ 70,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
        [ 80,       38,   50,   0.3,     70,   0.4672,   1,  28,   25],
        [ 90,       38,   50,   0.3,     70,   0.4672,   1,  28,   20],
        [100,       38,   50,   0.3,     70,   0.4672,   1,  28,   20]]   
    
    for i in range(0, 10):
        
        if chuveiro_sim.halted():
            break
            
        action = {
            'hot_valve_opening': TU[i][3],
            'out_valve_opening': TU[i][-4],
            'setpoint_final_temperature': TU[i][1],
            'gas_boiler_fraction': TU[i][2]
        }
            
        chuveiro_sim.episode_step(action)
        state = chuveiro_sim.get_state()
        print(state)
    
if __name__ == "__main__":
    main_test()
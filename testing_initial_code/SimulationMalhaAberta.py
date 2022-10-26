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
        
        self.time = 1
        
        # Definindo TU:
        # Time, Sr, Sa, xq, xf, xs, Fd, Td, Tinf
        TU = np.array(
        [   
              [0, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf],
              [self.time, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf]
        ])

        # Simulação malha aberta: 
        malha_aberta = MalhaAberta(ChuveiroTurbinado, self.T0, TU, dt = 0.01)
        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas
        self.TT, self.YY, self.UU = malha_aberta.solve_system()
        
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
        self.iqb = malha_aberta.compute_iqb(self.YY[:,-1], # T4a
                                             self.UU[:,4], # xs
                                             self.TT)

        # Cálculo do custo do banho:
        self.custo_eletrico, self.custo_gas = malha_aberta.custo_banho(self.UU[:,0], # Sr
                                                                        self.UU[:,2], # xq
                                                                        self.UU[:,3], # xf
                                                                        self.YY[:,2], # Tq
                                                                        self.UU[:,7], # Tinf
                                                                        self.TT,
                                                                        self.dt)

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
            T0 = config.get('initial_conditions')
        )
        
    def step(self):
        
        self.time += 10
        
        self.TU = np.append(self.TU, np.array([[self.time, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_aberta = MalhaAberta(ChuveiroTurbinado, self.T0, self.TU, dt = 0.01)
        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas
        self.TT, self.YY, self.UU = malha_aberta.solve_system()

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
        self.iqb = malha_aberta.compute_iqb(self.YY[:,-1], # T4a
                                             self.UU[:,4], # xs
                                             self.TT)

        # Cálculo do custo do banho:
        self.custo_eletrico, self.custo_gas = malha_aberta.custo_banho(self.UU[:,0], # Sr
                                                                        self.UU[:,2], # xq
                                                                        self.UU[:,3], # xf
                                                                        self.YY[:,2], # Tq
                                                                        self.UU[:,7], # Tinf
                                                                        self.TT,
                                                                        self.dt)
        
    # def episode_step(self, action: Schema) -> None:
    def episode_step(self, action) -> None:
        
        self.xq = action.get('hot_valve_opening')
        self.xf = action.get('cold_valve_opening')
        self.xs = action.get('out_valve_opening')
        self.Sr = action.get('electrical_resistence_fraction')
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
            'final_temperature': self.T4a,
            'final_temperature_boiler': self.Tq,
            'final_temperature_tank': self.T3,
            'tank_level': self.h,
            'quality_of_shower': self.iqb,
            'electrical_cost_shower': self.custo_eletrico,
            'gas_cost_shower': self.custo_gas 
        }
    
    def halted(self) -> bool:
        
        # Colocar constraints para nível do tanque h
        if self.T4a > 63 or self.h < 0 or self.h > 100:
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
    
         # Time,  Sr,   Sa,   xq,   xf,   xs,  Fd,  Td,  Tinf
    TU =[[  0,    50,   50,   0.3,  0.5,   1,   0,  25,   25],
         [  10,   40,   60,   0.3,  0.5,   1,   0,  25,   25],
         [  20,   50,   50,   0.7,  0.5,   1,   0,  25,   25],
         [  30,   50,   70,   0.7,  0.4,   1,   0,  25,   25],
         [  40,   60,   50,   0.7,  0.4,   1,   0,  25,   25],
         [  50,   50,   50,   0.7,  0.4,   1,   0,  25,   25]]


    for i in range(0, 5):
        
        if chuveiro_sim.halted():
            break
            
        action = {
            'hot_valve_opening': TU[i][3],
            'cold_valve_opening': TU[i][4],
            'out_valve_opening': TU[i][-4],
            'electrical_resistence_fraction': TU[i][1],
            'gas_boiler_fraction': TU[i][2]
        }
            
        chuveiro_sim.episode_step(action)
        state = chuveiro_sim.get_state()
        print(state)
    
if __name__ == "__main__":
    main_test()
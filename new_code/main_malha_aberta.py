import json
import os
import random
import numpy as np
import copy
import time as tm
import matplotlib.pyplot as plt

from bonsai_common import SimulatorSession, Schema
#import dotenv
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

from chuveiro_malha_aberta import MalhaAberta, ChuveiroTurbinado

# class ChuveiroTurbinadoSimulation(SimulatorSession):
class ChuveiroTurbinadoSimulation():
    
    def reset(
        self,
        Sr_0: float = 100,
        Sa_0: float = 100,
        xq_0: float = 0.3,
        xf_0: float = 0.3,
        xs_0: float = 0.4672,
        Fd_0: float = 0,
        Td_0: float = 25,
        Tinf: float = 25,
        T0: list = [50,  40,  40,  37],
        potencia_eletrica_0: float = 5.5,
        potencia_aquecedor_0: float = 29000,
        custo_eletrico_kwh_0: float = 1,
        custo_gas_kg_0: float = 2,
        custo_agua_m3_0: float = 3,
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
        potencia_eletrica_0: potência elétrica do chuveiro.
        potencia_aquecedor_0: potência do aquecedor a gás.
        custo_eletrico_kwh_0: custo elétrico em reais/kwh.
        custo_gas_kg_0: custo do gás em reais/kg.
        custo_agua_m3_0: custo da água em reais/m3.
        """
        
        # Variáveis iniciais:
        self.Sr = Sr_0
        self.Sa = Sa_0
        self.xq = xq_0
        self.xf = xf_0
        self.xs = xs_0
        self.Fd = Fd_0
        self.Td = Td_0
        self.Tinf = Tinf
        self.T0 = T0
        self.potencia_eletrica = potencia_eletrica_0
        self.potencia_aquecedor = potencia_aquecedor_0
        self.custo_eletrico_kwh = custo_eletrico_kwh_0
        self.custo_gas_kg = custo_gas_kg_0
        self.custo_agua_m3 = custo_agua_m3_0

        # Definição do tempo de duração de cada episódio em minutos:
        self.time = 2
        self.time_sample = 2
        
        # Definição do passo de tempo de simulação:
        self.dt = 0.01
        
        # Definição das variáveis iniciais: tempo, Sr, Sa, xq, SP(h), xs, Fd, Td, Tinf
        TU = np.array(
        [   
              [0, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf],
              [self.time, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf]
        ])

        # Simulação malha aberta: 
        malha_aberta = MalhaAberta(ChuveiroTurbinado, self.T0, TU, dt = 0.01)

        # Solução do sistema: 
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
        self.iqb = malha_aberta.calculo_iqb(self.T4a,
                                            self.xs)
        if np.isnan(self.iqb) or self.iqb == None or np.isinf(abs(self.iqb)):
            self.iqb = 0

        # Cálculo do custo elétrico do banho:
        self.custo_eletrico = malha_aberta.custo_eletrico_banho(self.Sr, 
                                                                self.potencia_eletrica, 
                                                                self.custo_eletrico_kwh,
                                                                self.time_sample)

        # Cálculo do custo de gás do banho:
        self.custo_gas = malha_aberta.custo_gas_banho(self.Sa,
                                                      self.potencia_aquecedor,
                                                      self.custo_gas_kg,
                                                      self.time_sample)

        # Cálculo do custo da água:
        self.custo_agua = malha_aberta.custo_agua(self.xs,
                                                  self.custo_agua_m3,
                                                  self.time_sample)

        # Cálculo da vazão final Fs para definiçao do estado:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))

        # Salva o estado atual:
        self.TU = TU
        self.TU_list = copy.deepcopy(TU)
        self.last_TU = copy.deepcopy(np.array([[self.time, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])

    # def episode_start(self, config: Schema) -> None:
    def episode_start(self, config) -> None:
        
        self.reset(
            Sr_0 = config.get('fracao_inicial_resistencia_eletrica') or 50,
            Sa_0 = config.get('fracao_inicial_aquecimento_boiler') or 50,
            xq_0 = config.get('abertura_inicial_valvula_quente') or 0.3,
            xf_0 = config.get('abertura_inicial_valvula_fria') or 0.5,
            xs_0 = config.get('abertura_inicial_valvula_saida') or 0.4672,
            Fd_0 = config.get('vazao_inicial_corrente_disturbio') or 0,
            Td_0 = config.get('temperatura_disturbio_inicial') or 25,
            Tinf = config.get('temperatura_ambiente_inicial') or 25, 
            # T0 = config.get('variaveis_estado_iniciais') or [50,  30 ,  30,  30],
            potencia_eletrica_0 = config.get('potencia_eletrica_inicial') or 5.5,
            potencia_aquecedor_0 = config.get('potencia_aquecedor_0') or 29000,
            custo_eletrico_kwh_0 = config.get('custo_eletrico_kwh_0') or 1,
            custo_gas_kg_0 = config.get('custo_gas_kg_0') or 2,
            custo_agua_m3_0 = config.get('custo_agua_m3_0') or 3,
        )
        
    def step(self):
        
        # Próximo episódio no tempo:
        self.time += self.time_sample 

        # Atribuindo valores das variáveis atuais:
        self.last_TU[0][1] = self.Sr
        self.last_TU[0][2] = self.Sa
        self.last_TU[0][3] = self.xq
        self.last_TU[0][4] = self.xf
        self.last_TU[0][5] = self.xs
        self.last_TU[0][6] = self.Fd 
        self.last_TU[0][7] = self.Td
        self.last_TU[0][8] = self.Tinf

        # Atribuindo valores para as próximas variáveis:
        self.TU = np.append(self.last_TU, np.array([[self.time, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        self.T0 = self.last_T0 # h, T3, Tq, T4a
        self.TU_list = np.append(self.TU_list, np.array([[self.time, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        # print(self.TU)
        # print(self.T0)
        # print(self.time)
        # print(self.TU_list)

        # Simulação malha aberta: 
        malha_aberta = MalhaAberta(ChuveiroTurbinado, self.T0, self.TU, dt = 0.01)

        # Solução do sistema: 
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
        self.iqb = malha_aberta.calculo_iqb(self.T4a,
                                            self.xs)
        if np.isnan(self.iqb) or self.iqb == None or np.isinf(abs(self.iqb)):
            self.iqb = 0

        # Cálculo do custo elétrico do banho:
        self.custo_eletrico = malha_aberta.custo_eletrico_banho(self.Sr, 
                                                                self.potencia_eletrica, 
                                                                self.custo_eletrico_kwh,
                                                                self.time_sample)

        # Cálculo do custo de gás do banho:
        self.custo_gas = malha_aberta.custo_gas_banho(self.Sa,
                                                      self.potencia_aquecedor,
                                                      self.custo_gas_kg,
                                                      self.time_sample)

        # Cálculo do custo da água:
        self.custo_agua = malha_aberta.custo_agua(self.xs,
                                                  self.custo_agua_m3,
                                                  self.time_sample)

        # Vazão final Fs:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))

        # Salvar o estado atual:
        self.last_TU = copy.deepcopy(np.array([[self.time, self.Sr, self.Sa, self.xq, self.xf, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])
        
    # def episode_step(self, action: Schema) -> None:
    def episode_step(self, action) -> None:
        
        self.xq = action.get('abertura_valvula_quente')
        self.xf = action.get('abertura_valvula_fria')
        self.xs = action.get('abertura_valvula_saida')
        self.Sr = action.get('fracao_resistencia_eletrica')
        self.Sa = action.get('fracao_aquecimento_boiler')
        self.Fd = action.get('vazao_corrente_disturbio')
        self.Td = action.get('temperatura_disturbio')
        self.Tinf = action.get('temperatura_ambiente')
        self.potencia_eletrica = action.get('potencia_eletrica')
        self.potencia_aquecedor = action.get('potencia_aquecedor')

        self.step()

    def get_state(self):
        
        return {  
            #'fracao_resistencia_eletrica': self.Sr,
            #'fracao_aquecimento_boiler': self.Sa,
            #'abertura_valvula_quente': self.xq,
            #'abertura_valvula_fria': self.xf,
            #'abertura_valvula_saida': self.xs,
            #'vazao_corrente_disturbio': self.Fd,
            #'temperatura_disturbio': self.Td,
            #'temperatura_ambiente': self.Tinf,
            'nivel_tanque': self.h,
            'temperatura_saida': self.T4a,
            'vazao_saida': self.Fs,
            'temperatura_final_boiler': self.Tq,
            'temperatura_final_tanque': self.T3,
            'qualidade_banho': self.iqb,
            'custo_eletrico_banho': self.custo_eletrico,
            'custo_gas_banho': self.custo_gas,
            'custo_agua_banho': self.custo_agua,
        }
    
    def halted(self) -> bool:
        
        # Colocar constraints para nível do tanque h
        if self.T4a > 63 or self.h < 0 or self.h > 100:
            return True
        else:
            return False

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

def main():

    workspace = os.getenv('SIM_WORKSPACE')
    access_key = os.getenv('SIM_ACCESS_KEY')

    # values in `.env`, if they exist, take priority over environment variables
    # dotenv.load_dotenv('.env', override=True)

    if workspace is None:
         raise ValueError('The Bonsai workspace ID is not set.')
    if access_key is None:
        raise ValueError('The access key for the Bonsai workspace is not set.')

    # config = BonsaiClientConfig(workspace=workspace, access_key=access_key)
    config = None

    chuveiro_sim = ChuveiroTurbinadoSimulation(config)

    chuveiro_sim.reset()

    while chuveiro_sim.run():
        continue
        
def main_test():
    
    configs_banho = {
        'potencia_eletrica': [5.5, 6.5, 7.5],
        'potencia_aquecedor': [25000, 27000, 29000],
        'custo_eletrico_kwh': [1, 1.5, 2], # reais/kwh
        'custo_gas_kg': [2, 3, 4], # reais/kg
        'custo_agua_m3': [3, 4, 5],
    }
    
    chuveiro_sim = ChuveiroTurbinadoSimulation()
    chuveiro_sim.reset(custo_eletrico_kwh_0 = random.choice(list(configs_banho['custo_eletrico_kwh'])),
                       custo_gas_kg_0 = random.choice(list(configs_banho['custo_gas_kg'])),
                       custo_agua_m3_0 = random.choice(list(configs_banho['custo_agua_m3'])))
    state = chuveiro_sim.get_state()

    q_list = []
    T4_list = []
    h_list = []
    Sr_list = []
    Sa_list = []
    time_list = []
    xq_list = []
    xf_list = [] 
    xs_list = [] 
    custo_eletrico_list = []
    custo_gas_list = []
    custo_agua_list = []
    Fd_list = []
    Td_list = []
    Tinf_list = []

    # Time, Sr, Sa, xq, xf, xs, Fd, Td, Tinf
    TU=[[20, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [30, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [40, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [50, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [60, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [70, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [80, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [90, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [100, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [110, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [120, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25],
        [130, 70, 100, 0.3, 0.1, 0.4672, 0, 25, 25]]   

    configs_banho = {
        'potencia_eletrica': [5.5, 6.5, 7.5],
        'potencia_aquecedor': [25000, 27000, 29000],
        'custo_eletrico_kwh': [1, 1.5, 2], # reais/kwh
        'custo_gas_kg': [2, 3, 4], # reais/kg
        'custo_agua_m3': [3, 4, 5],
    }
    
    for i in range(0, 12):
        
        if chuveiro_sim.halted():
            break
            
        action = {
            'abertura_valvula_quente': TU[i][3],
            'abertura_valvula_saida': TU[i][-4],
            'fracao_resistencia_eletrica': TU[i][1],
            'fracao_aquecimento_boiler': TU[i][2],
            'abertura_valvula_fria': TU[i][4],
            'vazao_corrente_disturbio': TU[i][6],
            'temperatura_disturbio': TU[i][7],
            'temperatura_ambiente': TU[i][8],
            'potencia_eletrica': random.choice(list(configs_banho['potencia_eletrica'])),
            'potencia_aquecedor': random.choice(list(configs_banho['potencia_aquecedor'])),
        }

        print('Ações:')
        print(action)
            
        chuveiro_sim.episode_step(action)
        state = chuveiro_sim.get_state()
        print('Estados')
        print(state)
        print('')
        q_list.append(state['qualidade_banho'])
        T4_list.append(state['temperatura_saida'])
        h_list.append(state['nivel_tanque'])
        Sr_list.append(action['fracao_resistencia_eletrica'])
        Sa_list.append(action['fracao_aquecimento_boiler'])
        time_list.append(TU[i][0])
        xq_list.append(action['abertura_valvula_quente'])
        xf_list.append(action['abertura_valvula_fria'])
        xs_list.append(action['abertura_valvula_saida'])
        custo_eletrico_list.append(state['custo_eletrico_banho'])
        custo_gas_list.append(state['custo_gas_banho'])
        custo_agua_list.append(state['custo_agua_banho'])
        Fd_list.append(action['vazao_corrente_disturbio'])
        Td_list.append(action['temperatura_disturbio'])
        Tinf_list.append(action['temperatura_ambiente'])

    # time_list = range(0, 12)
    plt.figure(figsize=(20, 15))
    plt.subplot(4,2,1)
    plt.plot(time_list, q_list, label='IQB')
    plt.legend()
    plt.subplot(4,2,2)
    plt.plot(time_list, T4_list, label='T4a')
    plt.legend()
    plt.subplot(4,2,3)
    plt.plot(time_list, h_list, label='h')
    plt.legend()
    plt.subplot(4,2,4)
    plt.plot(time_list, Sr_list, label='Sr')
    plt.plot(time_list, Sa_list, label='Sa')
    plt.legend()
    plt.subplot(4,2,5)
    plt.plot(time_list, xq_list, label='xq')
    plt.plot(time_list, xf_list, label='xf')
    plt.plot(time_list, xs_list, label='xs')
    plt.legend()
    plt.subplot(4,2,6)
    plt.plot(time_list, custo_eletrico_list, label='Custo elétrico')
    plt.plot(time_list, custo_gas_list, label='Custo gás')
    plt.plot(time_list, custo_agua_list, label='Custo água')
    plt.legend()
    plt.subplot(4,2,7)
    plt.plot(time_list, Td_list, label='Td')
    plt.plot(time_list, Tinf_list, label='Tinf')
    plt.legend()
    plt.subplot(4,2,8)
    plt.plot(time_list, Fd_list, label='Fd')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main_test()
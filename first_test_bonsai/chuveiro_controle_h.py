import math
import numpy as np
from scipy.integrate import solve_ivp

def ChuveiroTurbinado(t, 
                      z,                                   
                      Sr = 50, 
                      Sa = 100, 
                      xq = 0.5, 
                      xf = 0.5, 
                      xs = 0.5,  
                      Fd = 0, 
                      Td = 25, 
                      Tinf = 25):    
    
    """
    MODELO CHUVEIRO TURBINADO: condições de operação
    Vazão nominal da água de 5 gpm.
    Perda de carga na tubulção de saíde de 5 psi.
    Tanque e tubulação de saída estão à mesma pressão.
    Pressão estática de 20 psi na alimentação do tanque.
    Perda de carga de 1 psi para vazão nominal de 5 gpm da caixa d´água até o ponto de bifurcação entre a corrente quente e fria.
    Perda de carga no aquecedor de passagem para uma vazão nominal de 3 gpm é de 2 psi. 
    Vazão máxima de 10 gpm.
    A bomba centrífuga que opera a planta tem ΔP_Bmax = 30 psi e F_max = 20 gpm.
    Obturador linear na válvula com a corrente fria.
    Obturador parabólico na válvula com a corrente quente.
    Tanque cilíndrico com área da base A e volume do tanque dado por A * h, onde h é o nível do tanque.
    Propriedades do fluido e de aquecimento do sistema Q*R/p*c = 80.
    
    Parâmetros:
    t: tempo.
    z: variáveis de estado - h é o nível do tanque, T3 é a temperatura de saída do tanque de mistura de água fria e quente,
       Tq é a temperatura de saída do tanque de aquecimento com boiler e T4a é a temperatura de saída final.
    Sr: seletor que define a fração da resistência elétrica utilizada no aquecimento do tanque de mistura.
    Sa: seletor que define a fração para aquecimento à gás do boiler.
    xq: abertura da válvula para entrada da corrente quente aquecida pelo boiler.
    xf: abertura da válvula para entrada da corrente fria.
    xs: abertura da válvula para saída da corrente na tubulação final.
    Fd: vazão da corrente de distúrbio.
    Td: temperatura da corrente de distúrbio.
    Tinf: teperatura ambiente.
    """
    
    # Variáveis de estado (as que possuem as derivadas a resolver):
    # h = nível do tanque
    # T3 = temperatura de saída do tanque de mistura de água fria e quente
    # Tq = temperatura de saída do tanque de aquecimento com boiler
    # T4a = temperatura de saída final
    h, T3, Tq, T4a = z
    
    # Parâmetros do tanque:
    A = 0.5

    # Definindo Tf:
    Tf = Td

    # Implementação das vazões Fq, Ff e Fs:
    # Constantes para facilitar os cálculos conforme as condições de operação da planta
    t1 = xq ** 0.2
    t2 = t1 ** 2
    t5 = xf ** 2 
    t7 = xq ** 2 
    t9 = t1 * xq * t7
    t10 = t9 * t5
    t14 = t7 ** 2 
    t16 = t2 * t7 * t14 
    t17 = t16 * t5 
    t20 = np.sqrt(0.9e1 * t10 + 0.50e2 * t17) 
    t26 = t5 ** 2
    t38 = (np.sqrt(-0.1e1 * (-0.180e3 -0.45e2 * t5 -0.250e3 * t10 - 0.1180e4 * t9 + 0.60e2 * t20) / (0.6552e4 * t10 + 0.648e3 
           * t5 + 0.16400e5* t17 + 0.900e3 * t9 * t26 + 0.2500e4 * t16 * t26 + 0.81e2 * t26 + 0.55696e5 * t16 + 0.16992e5 
           * t9 + 0.1296e4)))
    
    # Equações para as vazões de corrente quente, fria e de saída da tubulação:
    Fq = 60 * t1 * t2 * xq * t38
    Ff = (2 * xf * np.sqrt(125 * xf ** 2 - Fq ** 2 + 500) - xf * Fq) / (xf ** 2 + 4)
    Fs = (5 * xs ** 3 * np.sqrt(30) * np.sqrt(-15 * xs ** 6 + np.sqrt(6625 * xs ** 12 + 640 * xs ** 6 + 16)) / (20 * xs** 6 + 1))
    
    # Equações diferenciais
    # Modelagem do aquecedor, do tanque de aquecimento com mistura entre correntes fria e quente e da tubulação de saída:
    return([(Fq + Ff + Fd - Fs) / A, 
            (Ff * (Tf - T3) + Fq * (Tq - T3) + Fd * (Td - T3) + Sr * 80 / 100)/(A * h),
            Fq * (Tf - Tq) + 50 * Sa / 100, 
            Fs / 2.5 * (T3 - T4a) - 0.8 * ((T3 - Tinf) * (T4a - Tinf) * (1 / 2 * (T3 - Tinf) + 1 / 2 * (T4a - Tinf))) ** (1 / 3)])
    
def PID_p2(SP, 
           PV, 
           k, 
           I_int, 
           D_int, 
           dt, 
           Method = 'Backward', 
           Kp = 10.0, 
           Ti = 50.0, 
           Td = 1.0, 
           b = 1.0, 
           c = 0.0, 
           N = 10.0, 
           U_bias = 0.0, 
           Umin = -100.0, 
           Umax = 100.0):
    
    """Controlador PID."""
    # Autor: Jorge Otávio Trierweiler -- jorge.trierweiler@ufrgs.br
    
    if Method == 'Backward':
        b1 = Kp * dt / Ti if Ti != 0 else 0.0
        b2 = 0.0
        ad = Td / (Td + N * dt)
        bd = Kp * Td * N / (Td + N * dt)
        
    elif Method == 'Forward':
        b1 = 0.0
        b2 = Kp * dt / Ti  if Ti != 0 else 0.0
        ad = 1 - N * dt / Td if Td != 0 else 0.0
        bd = Kp * N   
        
    elif Method == 'Tustin':
        b1 = Kp * dt / 2 / Ti if Ti != 0 else 0.0
        b2 = b1
        ad = (2 * Td - N * dt) / (2 * Td + N * dt)
        bd = 2 * Kp * Td * N / (2 * Td + N * dt) 
        
    elif Method == 'Ramp':
        b1 = Kp * dt / 2 / Ti if Ti != 0 else 0.0
        b2 = b1
        ad = np.exp(-N * dt / Td) if Td != 0 else 0.0
        bd = Kp * Td * (1 - ad) / dt
        
    # Ação derivativa:
    D  = ad * D_int + bd * ((c * SP[k] - PV[k]) - (c * SP[k-1] - PV[k-1]))
    
    # Ação integral
    II = b1 * (SP[k] - PV[k]) + b2 * (SP[k-1] - PV[k-1])
    I = I_int + II                         
     
    # Calcula o output do PID:
    P = Kp * (b * SP[k] - PV[k])
    Uop = U_bias + P + I + D

    # Implementa anti-reset windup:
    if Uop < Umin:
        II = 0.0     
        Uop = Umin
    if Uop > Umax:
        II = 0.0
        Uop = Umax
    
    # Retorna resultado do controlador e termos do PID:
    return np.array([Uop, I_int + II, D])

class MalhaFechada():
    
    def __init__(self,
                 SYS,
                 y0,
                 TU,
                 Kp_h = 2, 
                 Ti_h = 0.5, 
                 Td_h = 0.0, 
                 b_h = 1,
                 Ruido = 0.005, 
                 U_bias_h = 0.5, 
                 dt = 0.01):
        
        """
        Parâmetros:
        SYS: função do chuveiro turbinado com equações diferenciais a resolver.
        y0: condições iniciais para as variáveis de estado h, T3, Tq e T4a.
        TU: condições para as variáveis manipuladas (Sr, Sa, xq, xf, xs, Fd, Td, Tinf) em cada intervalo de tempo.
        dt: tempo de cada passo na simulação.
        
        Retorna:
        TT: tempo total com variação de dt em dt.
        YY: h = YY[:,0], T4a = YY[:,-1], T3 = YY[:,1]
        UU: Sr = UU[:,0], Sa = UU[:,1], xq = UU[:,2], xf = UU[:,3], xs = UU[:,4], Fd = UU[:,5], Td = UU[:,6], Tinf = UU[:,7]
        """
    
        self.SYS = SYS
        self.y0 = y0
        self.TU = TU
        self.Kp_h = Kp_h 
        self.Ti_h = Ti_h 
        self.Td_h = Td_h 
        self.b_h = b_h
        self.Ruido = Ruido 
        self.U_bias_h = U_bias_h 
        self.dt = dt 

    def solve_system(self):
        
        # Definição tempo final e condições iniciais da simulação:
        # Tfinal = self.TU[-1, 0] 
        Tfinal = self.TU[1][0]

        Yset_bias_h = self.y0[0]

        # Armazenamento dos dados resultantes da simulação:
        # TT = np.arange(start = 0, stop = Tfinal + self.dt, step = self.dt, dtype = 'float')
        TT = np.arange(start = self.TU[0][0], stop = Tfinal + self.dt, step = self.dt, dtype = 'float')
        NT = np.size(TT)
        NY = np.size(self.y0)
        YY = np.zeros((NT, NY))
        nu = np.size(self.TU, 1)
        UU = np.zeros((NT, nu-1))
        SP_h = np.zeros_like(TT)
        PV_h = np.ones_like(TT) * Yset_bias_h
        Noise = np.random.normal(0, self.Ruido, len(TT))

        YY[-1, 2] = self.y0[2]

        ii = 0    

        D_int_h = 0.0  
        I_int_h = self.Kp_h * Yset_bias_h * (1 - self.b_h)

        YY[0,:] = self.y0

        for k in np.arange(NT - 1):

            #print(TT[k])
            #print(self.TU[ii + 1, 0])

            if TT[k] >= self.TU[ii + 1, 0]:
                ii = ii + 1

            # Definição do setpoint e do distúrbio na carga:
            UU[k,:] = self.TU[ii, 1:nu]

            SP_h[k] = self.TU[ii, 4]
            PV_h[k] = YY[k, 0] + Noise[k]

            # Armazenamento dos valores calculados:
            # Malha nível:
            uu_h = PID_p2(SP_h, PV_h, k, I_int_h, D_int_h, self.dt, Method ='Backward',
                          Kp = self.Kp_h, Ti = self.Ti_h, Td = self.Td_h, N = 10, b = self.b_h, 
                          Umin = 0, Umax = 1, U_bias = self.U_bias_h)
            Uop_h = uu_h[0]
            I_int_h = uu_h[1]
            D_int_h = uu_h[2]

            # Definição do setpoint e do distúrbio na carga:
            UU[k, 3] = Uop_h

            sol = solve_ivp(self.SYS, [TT[k], TT[k+1]], YY[k,:], args = tuple(UU[k,:]), rtol = 1e-6) #,method="RK45",max_step = dt, atol=1, rtol=1)

            YY[k+1,:] = sol.y[:,-1]    

        # erro = yy_f-YY
        UU[k + 1,:] = self.TU[ii, 1:nu]

        UU[k + 1,3] = Uop_h
        SP_h[k + 1] = self.TU[ii, 4]

        # Resultados:
        return (TT, YY, UU)
    
    def calculo_iqb(self, T4a, xs):

        # Índice de qualidade do banho:
        Fs = (5 * xs ** 3 * np.sqrt(30) * np.sqrt(-15 * xs ** 6 + np.sqrt(6625 * xs ** 12 + 640 * xs ** 6 + 16)) / (20 * xs ** 6 + 1))
        IQB = (1 / math.e) * math.exp((1 - ((T4a - 38 + 0.02 * Fs) / 2) ** 2) * np.power((0.506 + math.log10(math.log10((10000 * np.sqrt(Fs)) / (10 + Fs + 0.004 * np.power(Fs, 4))))), 20))

        return IQB

    def custo_eletrico_banho(self, Sr, potencia_eletrica, custo_eletrico_kwh, time):

        # Custo da parte elétrica:
        custo_eletrico_total = potencia_eletrica * (Sr / 100) * custo_eletrico_kwh * time / 60

        return custo_eletrico_total

    def custo_gas_banho(self, Sa, potencia_aquecedor, custo_gas_kg, time):

        # Custo do gás:
        rendimento = 0.85
        potencia_util = potencia_aquecedor * rendimento # kcal/h
        potencia_final = potencia_util * (Sa / 100) # kcal/h

        kcal_fornecida_no_banho = potencia_final * time / 60
        kg_equivalente_kcal = 11750 # 1kg de gás equivale a 11750 kcal
        quantidade_gas_kg = kcal_fornecida_no_banho / kg_equivalente_kcal # kg
        custo_gas_total = custo_gas_kg * quantidade_gas_kg

        return custo_gas_total 

    def custo_agua(self, xs, custo_agua_m3, time):

        # Custo da água:
        Fs = (5 * xs ** 3 * np.sqrt(30) * np.sqrt(-15 * xs ** 6 + np.sqrt(6625 * xs ** 12 + 640 * xs ** 6 + 16)) / (20 * xs ** 6 + 1))
        custo_agua_total = Fs * (time / 1000) * custo_agua_m3 

        return custo_agua_total
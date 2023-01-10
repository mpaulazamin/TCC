inkling "2.0"
using Goal

const potencia_eletrica = 5.5 
const potencia_aquecedor = 290000 
const vazao_corrente_disturbio = 0
const temperatura_disturbio = 25


type SimConfig {
    fracao_inicial_resistencia_eletrica: number<0 .. 100>,
    fracao_inicial_aquecimento_boiler: number<0 .. 100>,
    abertura_inicial_valvula_quente: number<0 .. 0.4>,
    abertura_inicial_valvula_saida: number<0 .. 0.5>,
    setpoint_inicial_nivel_tanque: number<40 .. 70>,
    # vazao_inicial_corrente_disturbio: vazao_corrente_disturbio,
    # temperatura_disturbio_inicial: temperatura_disturbio,
    vazao_inicial_corrente_disturbio: number<0 .. 1>,
    temperatura_disturbio_inicial: number<20 .. 30 step 1>,
    temperatura_ambiente_inicial: number<20 .. 30 step 1>,
    potencia_eletrica_inicial: potencia_eletrica,
    potencia_aquecedor_inicial: potencia_aquecedor,
    custo_eletrico_kwh_inicial: number<1, 1.5, 2, 2.5, 3, 3.5>,
    custo_gas_kg_inicial: number<2, 2.5, 3>,
    custo_agua_m3_inicial: number<2, 3, 4, 5>
}

type SimState {
    # fracao_resistencia_eletrica: number,
    # fracao_aquecimento_boiler: number,
    # abertura_valvula_quente: number
    # abertura_valvula_fria: number,
    # abertura_valvula_saida: number,
    # vazao_corrente_disturbio: number,
    # temperatura_disturbio: number,
    # temperatura_ambiente: number,
    # setpoint_nivel_tanque: number,
    # nivel_tanque: number,
    # temperatura_saida: number,
    # vazao_saida: number,
    # temperatura_final_boiler: number,
    # temperatura_final_tanque: number,
    qualidade_banho: number,
    custo_eletrico_banho: number,
    custo_gas_banho: number,
    # custo_agua_banho: number,
}

type SimAction {
    abertura_valvula_quente: number <0 .. 0.4>,
    abertura_valvula_saida: number <0 .. 0.5>,
    fracao_resistencia_eletrica: number <0 .. 100>,
    fracao_aquecimento_boiler: number <0 .. 100>,
    setpoint_nivel_tanque: number <40 .. 70>,
}

graph (input: SimState): SimAction {

    concept DayBathNormalDay(input): SimAction {
        curriculum {
            source simulator(Action: SimAction, Config: SimConfig): SimState {
            }

            training {
                EpisodeIterationLimit: 1,
                TotalIterationLimit: 100
            }
            goal (state: SimState) {
                drive IQBIdeal:
                    state.qualidade_banho in Goal.Range(0.9, 1)
                minimize CustoEletrico:
                    state.custo_eletrico_banho in Goal.RangeBelow(1)
                maximize CustoGas:
                    state.custo_gas_banho in Goal.RangeAbove(0.5)
            }
            lesson LowDisturb {
                scenario {
                    custo_eletrico_kwh_inicial: number<1, 1.5, 2>,
                    temperatura_ambiente_inicial: number<25 .. 27>,
                    temperatura_disturbio_inicial: number<25 .. 27>,
                    vazao_inicial_corrente_disturbio: number<0 .. 0.5>,
                    fracao_inicial_resistencia_eletrica: number<60 .. 80>,
                    fracao_inicial_aquecimento_boiler: number<80 .. 100>
                }
            }
            lesson HighDisturb {
                scenario {
                    custo_eletrico_kwh_inicial: number<1, 1.5, 2>,
                    temperatura_ambiente_inicial: number<25 .. 30>,
                    temperatura_disturbio_inicial: number<25 .. 30>,
                    vazao_inicial_corrente_disturbio: number<0 .. 1>,
                    fracao_inicial_resistencia_eletrica: number<60 .. 80>,
                    fracao_inicial_aquecimento_boiler: number<80 .. 100>
                }
            }
        }
    }

    concept NightBathNormalDay(input): SimAction {
        curriculum {
            source simulator(Action: SimAction, Config: SimConfig): SimState {
            }
            training {
                EpisodeIterationLimit: 1,
                TotalIterationLimit: 100
            }
            goal (state: SimState) {
                drive IQBIdeal:
                    state.qualidade_banho in Goal.Range(0.9, 1)
                minimize CustoEletrico:
                    state.custo_eletrico_banho in Goal.RangeBelow(1)
                maximize CustoGas:
                    state.custo_gas_banho in Goal.RangeAbove(0.5)
            }
            lesson LowDisturb {
                scenario {
                    custo_eletrico_kwh_inicial: number<2.5, 3, 3.5>,
                    temperatura_ambiente_inicial: number<23 .. 25>,
                    temperatura_disturbio_inicial: number<23 .. 25>,
                    vazao_inicial_corrente_disturbio: number<0 .. 0.5>,
                    fracao_inicial_resistencia_eletrica: number<60 .. 80>,
                    fracao_inicial_aquecimento_boiler: number<80 .. 100>
                }
            }
            lesson HighDisturb {
                scenario {
                    custo_eletrico_kwh_inicial: number<2.5, 3, 3.5>,
                    temperatura_ambiente_inicial: number<20 .. 25>,
                    temperatura_disturbio_inicial: number<20 .. 25>,
                    vazao_inicial_corrente_disturbio: number<0 .. 1>,
                    fracao_inicial_resistencia_eletrica: number<60 .. 100>,
                    fracao_inicial_aquecimento_boiler: number<80 .. 100>
                }
            }
        }
    }

    output concept PickOne(input): SimAction {
        select DayBathNormalDay
        select NightBathNormalDay
        curriculum {
            source simulator(Action: SimAction, Config: SimConfig): SimState {
            }
            training {
                EpisodeIterationLimit: 1,
                TotalIterationLimit: 100
            }
            goal (state: SimState) {
                drive IQBIdeal:
                    state.qualidade_banho in Goal.Range(0.9, 1)
                minimize CustoEletrico:
                    state.custo_eletrico_banho in Goal.RangeBelow(1)
                maximize CustoGas:
                    state.custo_gas_banho in Goal.RangeAbove(0.5)
             }
             lesson One {
                 scenario {
                    custo_eletrico_kwh_inicial: number<1, 1.5, 2, 2.5, 3, 3.5>,
                    temperatura_ambiente_inicial: number<20 .. 30>,
                    fracao_inicial_resistencia_eletrica: number<60 .. 100>,
                    fracao_inicial_aquecimento_boiler: number<80 .. 100>
                 }
             }
        }
    }
}
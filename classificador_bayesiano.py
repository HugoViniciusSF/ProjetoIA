# Parâmetros para discretizar a contagem de carros
LEVE_THRESHOLD = 5          # Abaixo de LEVE_THRESHOLD carros na janela = Trânsito Leve
MODERADO_THRESHOLD = 12     # Entre LEVE_THRESHOLD e MODERADO_THRESHOLD carros = Trânsito Moderado
                            # Acima de MODERADO_THRESHOLD carros = Trânsito Alto

# Tabela de Probabilidade Condicional (CPT) 
CPT = {
    'BAIXA': {
        'NORMAL': {'LEVE': 0.90, 'MODERADO': 0.09, 'ALTO': 0.01},
        'PICO':   {'LEVE': 0.60, 'MODERADO': 0.35, 'ALTO': 0.05}
    },
    'MEDIA': {
        'NORMAL': {'LEVE': 0.20, 'MODERADO': 0.70, 'ALTO': 0.10},
        'PICO':   {'LEVE': 0.05, 'MODERADO': 0.65, 'ALTO': 0.30}
    },
    'ALTA': {
        'NORMAL': {'LEVE': 0.01, 'MODERADO': 0.39, 'ALTO': 0.60},
        'PICO':   {'LEVE': 0.01, 'MODERADO': 0.19, 'ALTO': 0.80}
    }
}

def classificar_estado_bayesiano(carros_contados, hora_atual):
    """Classifica o estado do trânsito usando a inferência Bayesiana."""
    contagem_cat = 'BAIXA' if carros_contados < LEVE_THRESHOLD else 'MEDIA' if carros_contados < MODERADO_THRESHOLD else 'ALTA'
    hora_cat = 'PICO' if (7 <= hora_atual < 9) or (17 <= hora_atual < 19) else 'NORMAL'
    
    probabilidades = CPT[contagem_cat][hora_cat]
    estado_mais_provavel = max(probabilidades, key=probabilidades.get)
    #print(probabilidades)
    return estado_mais_provavel
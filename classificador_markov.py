# Parâmetros para discretizar a contagem de veículos (os mesmos do Bayesiano para uma comparação justa)
LEVE_THRESHOLD = 5  # Abaixo de LEVE_THRESHOLD veículos na janela = Trânsito Leve
MODERADO_THRESHOLD = 12  # Entre LEVE_THRESHOLD e MODERADO_THRESHOLD veículos = Trânsito Moderado
# Acima de MODERADO_THRESHOLD veículos = Trânsito Alto

class ClassificadorMarkoviano:
    """Classifica o estado do trânsito usando uma Cadeia de Markov, que possui "memória" do estado anterior."""
    
    def __init__(self):
        self.current_state = "Indeterminado"
    
    def classificar_estado(self, veiculos_contados):
        """ Transita para um novo estado baseado no estado atual e na nova contagem total de veículos. """
        
        # Discretiza a evidência atual
        categoria_contagem = 'BAIXA' if veiculos_contados < LEVE_THRESHOLD else 'MEDIA' if veiculos_contados < MODERADO_THRESHOLD else 'ALTA'
        
        # Lógica de transição de estado (a "memória" de Markov)
        if self.current_state == "LEVE" or self.current_state == "Indeterminado":
            if categoria_contagem == 'ALTA':
                self.current_state = 'MODERADO'  # Não salta diretamente para ALTO
            elif categoria_contagem == 'MEDIA':
                self.current_state = 'MODERADO'
            else:  # BAIXA
                self.current_state = 'LEVE'
                
        elif self.current_state == "MODERADO":
            if categoria_contagem == 'ALTA':
                self.current_state = 'ALTO'
            elif categoria_contagem == 'BAIXA':
                self.current_state = 'LEVE'
            # Se for MEDIA, mantém-se em MODERADO
            
        elif self.current_state == "ALTO":
            if categoria_contagem == 'BAIXA':
                self.current_state = 'MODERADO'  # Não salta diretamente para LEVE
            elif categoria_contagem == 'MEDIA':
                self.current_state = 'MODERADO'
            # Se for ALTA, mantém-se em ALTO
            
        return self.current_state
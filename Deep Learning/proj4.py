import matplotlib
matplotlib.use('Agg') # Força backend não-interativo no Kubuntu
from ultralytics import YOLO
import os

def main():
    print("=== PROJETO 4: Deep Learning com Arquitetura YOLO ===")
    
    # 1. Definição da Arquitetura
    # Usamos o YOLOv8n (nano), que é uma CNN profunda otimizada.
    # Ele baixa automaticamente os pesos pré-treinados na primeira execução.
    print("Carregando arquitetura YOLOv8n...")
    model = YOLO("yolov8n.pt") 

    # 2. Configuração do Dataset
    # Certifique-se de que o arquivo 'data.yaml' está na mesma pasta
    yaml_path = os.path.abspath("trash-detection-kaggle/data.yaml")
    
    if not os.path.exists(yaml_path):
        print("ERRO: 'data.yaml' não encontrado.")
        return

    # 3. Treinamento (Deep Learning Training)
    print("Iniciando treinamento (Isso usa PyTorch por baixo dos panos)...")
    # epochs=10: Para testar rápido. No real, use 50 ou 100.
    # imgsz=640: Tamanho padrão do YOLO.
    results = model.train(
        data=yaml_path,
        epochs=10, 
        imgsz=640,
        plots=True,      # Gera gráficos automaticamente
        device='cpu'     # Force CPU se não tiver GPU Nvidia configurada
    )

    # 4. Validação
    print("Avaliando o modelo...")
    metrics = model.val()
    
    print(f"\nResultados Finais:")
    print(f"mAP50 (Precisão Média): {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # 5. Exportar para uso futuro
    path = model.export(format="onnx")
    print(f"Modelo exportado para: {path}")
    
    print("\nNOTA: Os gráficos de treinamento (perda, precisão) foram salvos")
    print("automaticamente na pasta 'runs/detect/train'")

if __name__ == '__main__':
    # Proteção necessária para multiprocessing no PyTorch/Linux
    main()
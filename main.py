import os
from dotenv import load_dotenv
from analisis import AnalizadorSentimiento

# Cargamos el .env para poder leer el umbral de seguridad
load_dotenv()

def iniciar_proceso():
    # 1. Instanciamos el motor (se carga en RAM una sola vez)
    motor = AnalizadorSentimiento()
    
    # 2. Leemos nuestro umbral desde el .env
    umbral_seguridad = float(os.getenv("CONFIDENCE_THRESHOLD", 80.0))
    print(f"\n[SISTEMA] Umbral de certeza configurado en: {umbral_seguridad}%\n")

    # 3. Casos de prueba
    tickets = [
        "El ejecutivo fue muy amable y resolvió mi problema de inmediato.",
        "El sistema arroja error al intentar subir el PDF, necesito ayuda urgente.",
        "Adjunto comprobante de pago de la cuota de este mes."
    ]

    print("--- Procesando Tickets ---\n")
    for ticket in tickets:
        resultado = motor.analizar_texto(ticket)
        print (f"Ticket: '{resultado}'")

if __name__ == "__main__":
    iniciar_proceso()
use ai_brain::TutorBrain;
use common::{MinaContext, EppItem};
use std::fs;
use std::path::Path;

fn main() {
    println!("=== Iniciando Test de Safety Buddy (AI Brain) ===");

    // 1. Ruta de tu modelo local (Asegúrate de cambiar esto a donde tengas tu modelo .gguf)
    // Recomendación: Llama 3 8B Instruct (Q4_K_M)
    let model_path = "assets/models/gemma-4-E4B-it-Q4_K_M.gguf"; 
    
    if !Path::new(model_path).exists() {
        eprintln!("ERROR: No se encontró el modelo en '{}'. Por favor ajusta la ruta.", model_path);
        return;
    }

    // 2. Inicializar el "Cerebro"
    println!("Cargando modelo en memoria (esto puede tomar unos segundos)...");
    let brain = match TutorBrain::new(model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Fallo al cargar el modelo: {}", e);
            return;
        }
    };
    println!("Modelo cargado exitosamente.\n");

    // 3. Simular los datos del Crate 'safety-engine' y 'common'
    let contexto = MinaContext {
        nombre_mina: "Mina Norte - Nivel Subterráneo".to_string(),
        riesgos_activos: vec!["Gases Tóxicos (CO, NO2)".to_string(), "Polvo en suspensión".to_string()],
        epp_obligatorio: vec![EppItem::Casco, EppItem::Respirador, EppItem::Lentes],
        path_protocolo: "../../protocols/norte_gases.md".to_string(),
    };

    // Simulamos que la cámara (visión) detectó el casco y los lentes, pero NO el respirador.
    let faltantes_simulados = vec![EppItem::Respirador];

    // 4. Cargar el texto del protocolo
    let protocolo_texto = fs::read_to_string(&contexto.path_protocolo)
        .unwrap_or_else(|_| String::from("ERROR: No se pudo leer el archivo de protocolo."));

    println!("Situación Simulada:");
    println!("- Mina: {}", contexto.nombre_mina);
    println!("- Riesgos: {:?}", contexto.riesgos_activos);
    println!("- EPP Faltante detectado: {:?}", faltantes_simulados);
    println!("\nConsultando al Tutor IA...\n");

    // 5. Evaluar la seguridad
    match brain.evaluate_safety(&contexto, &faltantes_simulados, &protocolo_texto) {
        Ok(respuesta) => {
            println!("=== RESPUESTA DEL TUTOR ===");
            println!("Es seguro ingresar: {}", respuesta.is_safe);
            println!("Severidad: {}", respuesta.severity);
            println!("Explicación: {}", respuesta.explanation);
            println!("===========================");
        }
        Err(e) => {
            eprintln!("Error durante la inferencia o parseo JSON:\n{}", e);
        }
    }
}
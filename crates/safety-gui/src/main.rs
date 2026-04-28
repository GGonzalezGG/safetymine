use eframe;
use anyhow::Result;

fn main() -> Result<(), eframe::Error> {
    // Configuración de la ventana egui
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("SafetyMine - Monitoreo en Tiempo Real"),
        ..Default::default()
    };

    let model_path = "best.onnx"; 
    
    // Usamos "0" como bandera para indicar que queremos la webcam por defecto
    let media_path = "0"; 

    println!("Iniciando SafetyMine Checkpoint...");
    println!("Cargando modelo YOLOv8 desde: {}", model_path);
    println!("Inicializando fuente de video: {}", media_path);

    eframe::run_native(
        "SafetyMine Checkpoint",
        options,
        Box::new(move |cc| {
            Box::new(vision::SafetyGuiApp::new(cc, model_path, media_path))
        }),
    )
}
use slint::{Model, VecModel, SharedString, Image, Rgba8Pixel, SharedPixelBuffer};
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::thread;
use std::time::Duration;
use image::DynamicImage;

use vision::{VisionEngine, CameraStream};
use safety_engine::SafetyEvaluator;
use common::MinaContext;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    println!("Iniciando SafetyMine GUI...");
    
    // 1. Cargar la DB inicial
    let db_path = "data/db.json"; 
    let db_content = std::fs::read_to_string(db_path).expect("No se pudo cargar db.json");
    let db: serde_json::Value = serde_json::from_str(&db_content).expect("Error en JSON");
    
    let nombres: Vec<SharedString> = db["minas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["nombre_mina"].as_str().unwrap().into())
        .collect();

    let primer_id = nombres[0].to_string();
    let evaluator = Arc::new(Mutex::new(SafetyEvaluator::new(db_path, &primer_id)));

    let ui = SafetyWindow::new()?;
    ui.set_lista_minas(Rc::new(VecModel::from(nombres)).into());

    // --- LÓGICA: Selección de Mina ---
    let ui_handle_mina = ui.as_weak();
    let evaluator_mina = evaluator.clone();
    
    ui.on_mina_seleccionada(move |nombre| {
        let mut ev = evaluator_mina.lock().unwrap();
        *ev = SafetyEvaluator::new("data/db.json", nombre.as_str());
        
        if let Some(ui) = ui_handle_mina.upgrade() {
            if let Some(ctx) = &ev.contexto_actual {
                let riesgos: Vec<SharedString> = ctx.riesgos_activos.iter().map(|r| r.as_str().into()).collect();
                ui.set_riesgos_mina(Rc::new(VecModel::from(riesgos)).into());

                let epp: Vec<SharedString> = ctx.epp_obligatorio.iter().map(|e| format!("{:?}", e).into()).collect();
                ui.set_lista_epp(Rc::new(VecModel::from(epp)).into());
            }
        }
    });

    // 2. Inicializamos YOLO y Estado Compartido
    println!("[Sistema] Cargando modelo ONNX...");
    let engine = Arc::new(VisionEngine::new("best.onnx").expect("Fallo al cargar YOLO"));
    
    // "ultimo_frame" será lo que procese YOLO, sin importar si vino de cámara o archivo
    let ultimo_frame = Arc::new(Mutex::new(None::<DynamicImage>));
    
    // "ultimo_frame" será lo que procese YOLO, sin importar si vino de cámara o archivo
    let ultimo_frame = Arc::new(Mutex::new(None::<DynamicImage>));
    
    // NUEVA VARIABLE: Controla el Switch entre cámara y archivo
    let modo_fuente = Arc::new(Mutex::new(String::new())); // Vacío = Cámara

    // ==================================================
    // LÓGICA 1: BOTÓN VOLVER A LA CÁMARA
    // ==================================================
    // Clonamos ANTES de entrar al botón
    let modo_fuente_cam_btn = modo_fuente.clone();
    
    ui.on_usar_camara(move || {
        println!("[UI] Volviendo a la cámara en vivo...");
        *modo_fuente_cam_btn.lock().unwrap() = String::new(); // Vaciamos la ruta
    });

    // ==================================================
    // LÓGICA 2: BOTÓN CAMBIAR FUENTE (RFD)
    // ==================================================
    let ui_handle_fuente = ui.as_weak();
    let ultimo_frame_rfd = ultimo_frame.clone();
    let modo_fuente_rfd = modo_fuente.clone();
    
    ui.on_cambiar_fuente(move || {
        let ui_weak = ui_handle_fuente.clone();
        let ultimo_frame_bg = ultimo_frame_rfd.clone();
        let modo_bg = modo_fuente_rfd.clone();
        
        thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Imágenes", &["jpg", "jpeg", "png", "webp"])
                .set_title("Seleccionar imagen de prueba")
                .pick_file() 
            {
                let ruta_completa = path.display().to_string();
                let nombre_archivo = path.file_name().unwrap().to_string_lossy().to_string();
                
                println!("[UI] Cargando imagen estática: {}", ruta_completa);
                
                // 1. Apagamos visualmente la cámara guardando la ruta del archivo
                *modo_bg.lock().unwrap() = ruta_completa.clone();

                // 2. Cargamos la imagen desde el disco
                if let Ok(img) = image::open(&path) {
                    *ultimo_frame_bg.lock().unwrap() = Some(img.clone());

                    let rgba = img.into_rgba8();
                    let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
                        rgba.as_raw(), rgba.width(), rgba.height()
                    );

                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let slint_img = Image::from_rgba8(buffer);
                            ui.set_frame_camara(slint_img);
                            ui.set_fuente_actual(format!("Archivo: {}", nombre_archivo).into());
                            
                            // Forzamos la validación automática
                            ui.invoke_validar_manual();
                        }
                    });
                }
            }
        });
    });

    // Forzar actualización inicial
    ui.invoke_mina_seleccionada(db["minas"][0]["nombre_mina"].as_str().unwrap().into());
    // --- HILO DE VIDEO EN VIVO ---
    let ui_handle_video = ui.as_weak();
    let ultimo_frame_cam = ultimo_frame.clone();
    let modo_fuente_cam = modo_fuente.clone();

    thread::spawn(move || {
        if let Ok(mut cam) = CameraStream::new(0) {
            loop {
                // Solo leemos de la cámara si no hay un archivo seleccionado
                let es_camara = modo_fuente_cam.lock().unwrap().is_empty();
                
                if es_camara {
                    if let Ok(img) = cam.get_frame() {
                        if let Ok(mut buz) = ultimo_frame_cam.lock() {
                            *buz = Some(img.clone());
                        }

                        let rgba = img.into_rgba8();
                        let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
                            rgba.as_raw(), rgba.width(), rgba.height()
                        );

                        let _ = slint::invoke_from_event_loop({
                            let ui_weak = ui_handle_video.clone();
                            move || {
                                if let Some(ui) = ui_weak.upgrade() {
                                    ui.set_frame_camara(Image::from_rgba8(buffer));
                                    ui.set_fuente_actual("Cámara Web (En Vivo)".into());
                                }
                            }
                        });
                    }
                }
                thread::sleep(Duration::from_millis(33)); 
            }
        }
    });

    // --- BOTÓN VALIDAR ---
    let engine_clone = engine.clone();
    let ultimo_frame_btn = ultimo_frame.clone();
    let evaluator_btn = evaluator.clone();

    ui.on_validar_manual(move || {
        println!("\n--- INICIANDO PRUEBA DE VISIÓN Y SEGURIDAD ---");
        let frame_actual = ultimo_frame_btn.lock().unwrap().clone();

        match frame_actual {
            Some(img) => {
                match engine_clone.process_frame(&img) {
                    Ok(detecciones) => {
                        println!("[1/2] YOLO finalizado. Detecciones: {}", detecciones.len());
                        for det in &detecciones {
                            println!("  -> {:?} (Confianza: {:.1}%)", det.label, det.score * 100.0);
                        }

                        // CRUZAMOS LOS DATOS DE VISIÓN CON LA BASE DE DATOS
                        println!("[2/2] Evaluando cumplimiento EPP...");
                        let ev = evaluator_btn.lock().unwrap();
                        
                        // Envolvemos las detecciones en la estructura que pide tu safety-engine
                        let resultado_vision = common::DetectionResult { 
                            detecciones,
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(), // Genera un timestamp real en números
                        };
                        let faltantes = ev.verificar_faltantes(&resultado_vision);
                        
                        if faltantes.is_empty() {
                            println!("  ✅ Cumplimiento total. El trabajador tiene todos los EPP.");
                        } else {
                            println!("  ❌ PELIGRO: Faltan EPP obligatorios:");
                            for f in faltantes {
                                println!("     - {:?}", f);
                            }
                        }
                    }
                    Err(e) => println!("Error procesando frame en YOLO: {}", e),
                }
            }
            None => println!("Aún no hay imagen cargada."),
        }
        println!("----------------------------------------------\n");
    });

    ui.run()
}
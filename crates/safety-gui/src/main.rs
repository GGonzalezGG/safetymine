use slint::{Model, VecModel};
use std::rc::Rc;

// Este macro compila el archivo .slint y genera el código Rust asociado
slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    println!("Iniciando SafetyMine GUI (Fase 1)...");
    
    // Inicializar la ventana
    let ui = SafetyWindow::new()?;

    // --- EVENTO: Cambiar Fuente ---
    let ui_handle = ui.as_weak();
    ui.on_cambiar_fuente(move || {
        println!("[Rust] Abriendo diálogo de archivos (Implementación en Fase 2)");
        if let Some(ui) = ui_handle.upgrade() {
            ui.set_fuente_actual("Archivo: test_video.mp4".into());
        }
    });

    // --- EVENTO: Selección de Mina ---
    ui.on_mina_seleccionada(|mina| {
        println!("[Rust] Recargando base de datos para: {}", mina);
    });

    // --- EVENTO: Botón Validar (Simulando una alerta del LLM) ---
    let ui_handle_alert = ui.as_weak();
    ui.on_validar_manual(move || {
        println!("[Rust] Procesando validación...");
        
        if let Some(ui) = ui_handle_alert.upgrade() {
            // Cambiamos el estado visual a ROJO
            ui.set_estado_general("RIESGO CRÍTICO DETECTADO".into());
            ui.set_color_estado(slint::Color::from_rgb_u8(244, 67, 54)); // Rojo Material
            
            // Alteramos la lista de EPP (Simulando que falta algo)
            let nuevo_epp = Rc::new(VecModel::from(vec![
                "✅ Casco".into(), 
                "❌ Respirador (FALTANTE)".into()
            ]));
            ui.set_lista_epp(nuevo_epp.into());

            // Agregamos un mensaje falso del Tutor al historial
            let mut historial_actual: Vec<slint::SharedString> = ui.get_historial_tutor().iter().collect();
            historial_actual.insert(0, "TUTOR: Detecto que no llevas tu máscara 3M. Según el protocolo de la Mina Norte, hay alta concentración de polvo. Sin ella podrías sufrir asfixia.".into());
            
            // Limitamos a 3 mensajes
            if historial_actual.len() > 3 {
                historial_actual.truncate(3);
            }
            
            let nuevo_historial = Rc::new(VecModel::from(historial_actual));
            ui.set_historial_tutor(nuevo_historial.into());
        }
    });

    // Arrancar el bucle principal de la UI
    ui.run()
}
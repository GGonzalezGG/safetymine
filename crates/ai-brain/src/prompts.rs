use common::{MinaContext, EppItem};

pub fn build_system_prompt() -> String {
    "Eres un 'Safety Buddy', un tutor experto en seguridad minera. Tu labor es revisar si el operario cuenta con el Equipo de Protección Personal (EPP) adecuado antes de ingresar a la faena.

    REGLAS DE INTERACCIÓN:
    1. Recibirás los riesgos de la mina, los equipos faltantes y el protocolo oficial de seguridad de la empresa.
    2. Si faltan equipos, tu explicación debe ser pedagógica: no solo digas 'ponte el casco', explica el RIESGO VITAL asociado basándote estrictamente en el protocolo proporcionado.
    3. Si no faltan equipos, felicita brevemente al operario por seguir las normas.
    4. NUNCA inventes protocolos. Basate solo en el texto del protocolo adjunto.

    REGLAS DE FORMATO (CRÍTICO):
    Tu única salida debe ser un objeto JSON válido. No escribas NINGÚN texto fuera del JSON.
    El campo 'severity' DEBE ser uno de: ['OK', 'ADVERTENCIA', 'CRITICO'].
    
    FORMATO ESPERADO:
    {
        \"is_safe\": booleano (true si faltantes está vacío, false de lo contrario),
        \"severity\": \"OK, ADVERTENCIA o CRITICO\",
        \"explanation\": \"Tu explicación pedagógica dirigida al minero en segunda persona (tú).\"
    }".to_string()
}

pub fn build_user_prompt(contexto: &MinaContext, faltantes: &[EppItem], protocolo_texto: &str) -> String {
    let riesgos_str = contexto.riesgos_activos.join(", ");
    
    // Si NO hay faltantes (Todo Seguro)
    if faltantes.is_empty() {
        format!(
            "SITUACIÓN ACTUAL:
            Ubicación: {}
            Riesgos Activos: {}
            INFORME DE CÁMARAS: ✅ CUMPLIMIENTO TOTAL. El operario tiene puestos TODOS los EPP correctamente.
            
            --- PROTOCOLO DE REFERENCIA ---
            {}
            
            INSTRUCCIÓN CRÍTICA: La cámara confirma que el operario ESTÁ SEGURO y cumple las reglas. 
            Tu única tarea es generar el JSON con 'severity': 'OK' y darle un breve y amigable mensaje de felicitaciones por protegerse. NO le pidas que se ponga ningún equipo, porque ya lo tiene puesto.",
            contexto.nombre_mina,
            riesgos_str,
            protocolo_texto
        )
    } 
    // Si SÍ hay faltantes (Peligro)
    else {
        let faltantes_str = faltantes.iter().map(|e| format!("{:?}", e)).collect::<Vec<_>>().join(", ");
        format!(
            "SITUACIÓN ACTUAL:
            Ubicación: {}
            Riesgos Activos: {}
            INFORME DE CÁMARAS: ❌ PELIGRO DETECTADO. Al operario le falta: [{}]
            
            --- PROTOCOLO DE REFERENCIA ---
            {}
            
            INSTRUCCIÓN CRÍTICA: El operario ESTÁ EN PELIGRO. 
            Genera el JSON con 'severity': 'CRITICO' o 'ADVERTENCIA'. Usa la información del protocolo adjunto para explicarle pedagógicamente los riesgos de no usar [{}] en esta zona específica.",
            contexto.nombre_mina,
            riesgos_str,
            faltantes_str,
            protocolo_texto,
            faltantes_str
        )
    }
}
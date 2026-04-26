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
    
    let faltantes_str = if faltantes.is_empty() {
        "Ninguno. El operario lleva todo el equipo.".to_string()
    } else {
        faltantes.iter().map(|e| format!("{:?}", e)).collect::<Vec<_>>().join(", ")
    };

    format!(
        "SITUACIÓN ACTUAL:
        Ubicación: {}
        Riesgos Activos: {}
        EPP Faltante detectado por las cámaras: {}
        
        --- INICIO DEL PROTOCOLO DE LA MINA ---
        {}
        --- FIN DEL PROTOCOLO ---
        
        Genera el JSON evaluando la situación actual para el operario.",
        contexto.nombre_mina,
        riesgos_str,
        faltantes_str,
        protocolo_texto
    )
}
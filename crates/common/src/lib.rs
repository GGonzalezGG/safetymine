use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum EppItem {
    Casco,
    Mascarilla,
    SinCasco,
    SinMascarilla,
    SinChaleco,
    Persona,
    ConoSeguridad,
    ChalecoSeguridad,
    Maquinaria,
    Vehiculo,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub label: EppItem,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetectionResult {
    pub detecciones: Vec<BoundingBox>,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MinaContext {
    pub nombre_mina: String,
    pub riesgos_activos: Vec<String>, // Ej: ["Gases", "Derrumbes"]
    pub epp_obligatorio: Vec<EppItem>,
    pub path_protocolo: String,       // Ruta al archivo .md
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIResponse {
    pub is_safe: bool,
    pub severity: String, // Ej: "OK", "ADVERTENCIA", "CRITICO"
    pub explanation: String,
}
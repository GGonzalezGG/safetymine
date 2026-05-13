fn main() {
    // Le decimos a Rust que compile nuestra interfaz visual
    slint_build::compile("ui/safety_window.slint").expect("Error compilando Slint");
}
/// Encode a `ShaderType` into a `Vec<u8>` laid out per WGSL's uniform-buffer
/// (std140) rules. encase enforces the alignment/padding at compile time, so we
/// no longer hand-roll padding fields or rely on `#[repr(C)]` matching the GPU.
pub fn uniform_bytes<T>(value: &T) -> Vec<u8>
where
    T: encase::ShaderType + encase::internal::WriteInto,
{
    let mut buffer = encase::UniformBuffer::new(Vec::<u8>::new());
    buffer
        .write(value)
        .expect("writing into an in-memory Vec buffer is infallible");
    buffer.into_inner()
}

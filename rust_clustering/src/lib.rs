use pyo3::prelude::*;

mod fennel;

/// L2G Clustering algorithms implemented in Rust for high performance
#[pymodule]
fn l2g_clustering(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fennel::fennel_clustering_rust, m)?)?;
    m.add_function(wrap_pyfunction!(fennel::fennel_clustering_parallel_rust, m)?)?;
    Ok(())
}
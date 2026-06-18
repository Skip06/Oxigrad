use std::collections::HashSet;
use crate::Value;

//Graphviz  (mirror Karpathy's .node() / .edge() calls) 

fn dot_value_node(id: usize, data: f64, grad: f64, label: &str) -> String {
    let lbl = if label.is_empty() {
        format!("{{ data={data:.4} | grad={grad:.4} }}")
    } else {
        format!("{{ {label} | data={data:.4} | grad={grad:.4} }}")
    };
    format!("  n{id} [label=\"{lbl}\", shape=record];\n")
}

fn dot_op_node(uid: &str, op: &str) -> String {
    format!("  {uid} [label=\"{op}\", shape=ellipse, style=filled, fillcolor=lightyellow];\n")
}

fn dot_edge(from: &str, to: &str) -> String {
    format!("  {from} -> {to};\n")
}

// traversal 

fn build_dot(
    value: &Value,
    nodes: &mut String,
    edges: &mut String,
    visited: &mut HashSet<usize>,
    label: &str,
) {
    let id = value.node_id();
    if visited.contains(&id) { return; }
    visited.insert(id);

    // Collect all data before dropping the borrow (avoids borrow-across-recurse)
    let (data, grad, op, children) = value.internals();

    // Value rectangle node
    nodes.push_str(&dot_value_node(id, data, grad, label));

    if !op.is_empty() {
        // Op ellipse node — unique id derived from the value's pointer
        let op_uid = format!("op_{id}");
        nodes.push_str(&dot_op_node(&op_uid, &op));
        // op  →  this value
        edges.push_str(&dot_edge(&op_uid, &format!("n{id}")));
        // each child  →  op
        for child in &children {
            let child_id = child.node_id();
            edges.push_str(&dot_edge(&format!("n{child_id}"), &op_uid));
        }
    }

    // Recurse into children (no label for non-root nodes)
    for child in &children {
        build_dot(child, nodes, edges, visited, "");
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Returns a Graphviz DOT string for the full compute graph rooted at `value`.
/// Pass a `root_label` like `"loss"` to annotate the root node, or `""` for none.
pub fn draw_dot(value: &Value, root_label: &str) -> String {
    let mut nodes = String::new();
    let mut edges = String::new();
    let mut visited = HashSet::new();
    build_dot(value, &mut nodes, &mut edges, &mut visited, root_label);
    format!(
        "digraph G {{\n  rankdir=LR;\n  node [fontname=\"Helvetica\"];\n{}{}}}",
        nodes, edges
    )
}

/// Saves the DOT file and renders it to SVG using the system `dot` binary.
pub fn render(value: &Value, root_label: &str, stem: &str) {
    use std::{fs, process::Command};

    let dot_path = format!("{stem}.dot");
    let svg_path = format!("{stem}.svg");

    let dot_str = draw_dot(value, root_label);
    fs::write(&dot_path, &dot_str).expect("could not write .dot file");

    let status = Command::new("dot")
        .args(["-Tsvg", &dot_path, "-o", &svg_path])
        .status();

    match status {
        Ok(_) => println!("\nCompute graph saved to {svg_path} — open it in a browser"),
        Err(_) => println!(
            "\n{dot_path} written. Install graphviz to render:\n  sudo apt install graphviz && dot -Tsvg {dot_path} -o {svg_path}"
        ),
    }
}

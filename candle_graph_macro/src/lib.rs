extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Make this struct an input for the graph, specifically, implement `GraphInput`.
///
/// ## Restrictions
/// - Only structs with named fields are supported.
/// - Each field's type must be of `candle_core::Tensor`.
///
/// ## Example
/// ```ignore
/// use candle_graph_macro::GraphInputItem;
/// use candle_core:Tensor;
///
/// #[derive(GraphInputItem)]
/// struct Person {
///     x: Tensor,
///     y: Tensor,
/// }
/// ```
#[proc_macro_derive(GraphInputItem)]
pub fn graph_input_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Get the name of the struct
    let name = &input.ident;

    // Generate code to match the struct's fields
    let st_data = if let Data::Struct(st_data) = input.data {
        st_data
    } else {
        panic!("`GraphInputItem` only works with structs.");
    };

    let fields = if let Fields::Named(named) = st_data.fields {
        named
            .named
            .iter()
            .map(|x| x.ident.clone().unwrap())
            .collect::<Vec<_>>()
    } else {
        panic!("`GraphInputItem` only works with structs that have named fields.");
    };

    let mut to_inputs_assigner = proc_macro2::TokenStream::new();
    quote_into::quote_into!(to_inputs_assigner += [#{
        for name in fields.clone() {
            quote_into::quote_into!(to_inputs_assigner += (inputs.push((stringify!(#name).to_string(), self.#name.clone()))),)
        }
    }];);

    let mut load_inputs_assigner = proc_macro2::TokenStream::new();
    quote_into::quote_into!(load_inputs_assigner += [#{
        for name in fields.clone() {
            quote_into::quote_into!(load_inputs_assigner += (unsafe { candle_graph::copy_inplace(&input.#name, &self.#name, device)? }),)
        }
    }];);

    let expanded = quote! {
        impl candle_graph::GraphInput for #name {
            fn to_inputs(&self) -> std::collections::HashMap<String, candle_core::Tensor> {
                let mut inputs = Vec::new();
                #to_inputs_assigner
                inputs.into_iter().collect()
            }
            fn load_inputs_inplace(&self, input: Self, device: &candle_core::Device) -> anyhow::Result<()> {
                #load_inputs_assigner
                Ok(())
            }
        }
    };

    TokenStream::from(expanded)
}

use tract_core::ndarray;
use tract_core::prelude::*;
use rand::*;

fn main() -> TractResult<()> {
    // load the model
    let model = tract_tensorflow::tensorflow().model_for_path("./my_model.pb")?;

    // Generate some input data for the model
    let mut rng = thread_rng();
    let vals: Vec<_> = (0..1000).map(|_| rng.gen::<f32>()).collect();
    let input = ndarray::arr1(&vals).into_shape((10, 100)).unwrap();

    // Make an execution plan for the model
    let plan = SimplePlan::new(&model).unwrap();

    // Input the generated data into the model
    let result = plan.run(tvec![input.into()]).unwrap();
    let to_show = result[0].to_array_view::<f32>()?;
    println!("result: {:?}", to_show);
    Ok(())
}
    

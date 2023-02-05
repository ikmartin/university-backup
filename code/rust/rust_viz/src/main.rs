use plotters::prelude::*;
use std::error::Error;
use fast_float::parse;

fn read_csv() -> Result<(Vec<f64>,Vec<i32>), Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_path("~/Desktop/university/code/rust/rust_viz/src/kc_house_data.csv")?;
    let mut price:Vec<f64> = Vec::new();
    let mut sqft_living:Vec<i32> = Vec::new();

    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here..
        let record = result?;
        match record.get(2) {
            Some(i) => {
                let tmp: f64 = parse(i).unwrap();
                price.push(tmp/1000.0)
            },
            _ => ()
        }
        match record.get(5) {
            Some(i) => {
                sqft_living.push(i.parse::<i32>().unwrap())
            },
            _ => ()
        }
    }
    
    return Ok((price, sqft_living));
}

fn main() {
    let house = match read_csv(){
        Ok(t) => t,
        _ => (Vec::new(),Vec::new())
    };
    let price = house.0;
    let sqft_living = house.1;

    let price_sqft_living: Vec<(f64, i32)>= price.iter().cloned().zip(sqft_living.iter().cloned()).collect();

    let root_area = BitMapBackend::new("~/Downloads/test.png", (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40.0)
        .set_label_area_size(LabelAreaPosition::Bottom, 40.0)
        .set_label_area_size(LabelAreaPosition::Right, 40.0)
        .set_label_area_size(LabelAreaPosition::Top, 40.0)
        .caption("House Sales in King County", ("sans-serif", 40.0))
        .build_cartesian_2d(0.0..8000.0, 0..10000)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    // Draw Scatter Plot
    ctx.draw_series(
        price_sqft_living.iter().map(|point| Circle::new(*point, 4.0_f64, &BLUE)),
    ).unwrap();

    // Draw Area Plot
    // ctx.draw_series(
    //     AreaSeries::new((0..).zip(sqft_living[..10].iter().map(|x| *x)), 0 ,&RED.mix(0.2)).border_style(&RED)
    // ).unwrap();

    // Draw Bar Plot
    // ctx.draw_series((0..).zip(sqft_living[..10].iter()).map(|(x, y)| {
    //     let x0 = SegmentValue::Exact(x);
    //     let x1 = SegmentValue::Exact(x + 1);
    //     let mut bar = Rectangle::new([(x0, 0), (x1, *y)], BLUE.filled());
    //     bar.set_margin(0, 0, 5, 5);
    //     bar
    // })).unwrap();

    // Draw Line Chart
    // ctx.draw_series(
    //     LineSeries::new((0..).zip(sqft_living[..10].iter()).map(|(idx, y)| {(idx, *y)}),&BLUE)
    // ).unwrap();
}

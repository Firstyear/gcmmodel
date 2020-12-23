use csv::Reader;
use serde::Deserialize;
// argmin

// WARNING: Field order matters in serde!
#[derive(Debug, Deserialize, PartialEq)]
struct Datum {
    embody_agg: f64,
    adv_agg: f64,
    creat_agg: f64,
    embed_agg: f64,
    authentic: f64,
    pred_comp: f64,
    identify: f64,
    r_effect: f64,
}

struct GCMCompetent {
    data: Vec<Datum>,
}

#[derive(Debug)]
struct Params {
    c: f64,
    gamma: f64,
    w: [f64; 6],
    b: [f64; 2],
}

impl Datum {
    fn manipulate(&mut self) {
        // In the example code, each field is -1 then div 6. Who knows why?
        self.embody_agg = (self.embody_agg - 1.0) / 6.0;
        self.adv_agg = (self.adv_agg - 1.0) / 6.0;
        self.creat_agg = (self.creat_agg - 1.0) / 6.0;
        self.embed_agg = (self.embed_agg - 1.0) / 6.0;
        self.authentic = (self.authentic - 1.0) / 6.0;
        self.identify = (self.identify - 1.0) / 6.0;
        self.r_effect = (self.r_effect - 1.0) / 6.0;
        self.pred_comp = (self.pred_comp - 1.0) / 6.0;
    }
}

fn import_datum_from_csv(path: &str) -> Vec<Datum> {
    let mut rdr = Reader::from_path(path).expect("Unable to access csv file");

    let res: Result<Vec<Datum>, _> = rdr
        .deserialize()
        // This is a iter of results.
        .map(|r| {
            // And we want to map ok values to Datum.
            r.map(|mut d: Datum| {
                d.manipulate();
                d
            })
        })
        .collect();

    res.expect("Encountered invalid data in csv")
}

impl GCMCompetent {
    fn new(data: Vec<Datum>) -> Self {
        GCMCompetent { data }
    }

    // Return peffective.
    fn predict(&self, parms: &Params) -> Vec<f64> {
        let wsum: f64 = parms.w.iter().sum();
        // Unrolled as [] isn't fromiterator
        let normWeight: [f64; 6] = [
            parms.w[0] / wsum,
            parms.w[1] / wsum,
            parms.w[2] / wsum,
            parms.w[3] / wsum,
            parms.w[4] / wsum,
            parms.w[5] / wsum,
        ];

        let neg_c = -(parms.c);

        println!("normWeight -> {:?}", normWeight);

        let peffective: Vec<f64> = self
            .data
            .iter()
            .enumerate()
            // This is the current row aka stim.
            .map(|(i, stim)| {
                // Todo: This is a prime candidate for SIMD!
                let dist: Vec<[f64; 6]> = self
                    .data
                    .iter()
                    .enumerate()
                    .filter_map(|(ki, d)| {
                        if ki == i {
                            // Exclude the current row,
                            debug_assert!(d == stim);
                            None
                        } else {
                            // abs(stim-exemplars);
                            // stim is
                            Some([
                                (stim.embody_agg - d.embody_agg).abs(),
                                (stim.adv_agg - d.adv_agg).abs(),
                                (stim.creat_agg - d.creat_agg).abs(),
                                (stim.embed_agg - d.embed_agg).abs(),
                                (stim.authentic - d.authentic).abs(),
                                (stim.identify - d.identify).abs(),
                            ])
                        }
                    })
                    .collect();
                debug_assert!(dist.len() == self.data.len() - 1);

                // inter.*dist
                // element wise multiply
                //
                // [ A B ]    [ E F ]    [ A*E B*F ]
                // [ C D ] .* [ G H ] == [ C*G D*H ]
                let inter_b: Vec<[f64; 6]> = dist
                    .into_iter()
                    .map(|row| {
                        [
                            row[0] * normWeight[0],
                            row[1] * normWeight[1],
                            row[2] * normWeight[2],
                            row[3] * normWeight[3],
                            row[4] * normWeight[4],
                            row[5] * normWeight[5],
                        ]
                    })
                    .collect();

                let wdist: Vec<f64> = inter_b.into_iter().map(|row| row.iter().sum()).collect();

                let sim: Vec<f64> = wdist.into_iter().map(|w| (neg_c * w).exp()).collect();

                let (sum_sim_a, sum_sim_b): (f64, f64) = sim
                    .into_iter()
                    .zip(self.data.iter().enumerate().filter_map(|(ki, d)| {
                        if ki == i {
                            // Exclude the current row,
                            debug_assert!(d == stim);
                            None
                        } else {
                            Some(d.r_effect)
                        }
                    }))
                    .fold((0.0, 0.0), |(acc_a, acc_b), (sim, cat)| {
                        (acc_a + (sim * cat), acc_b + (sim * (1.0 - cat)))
                    });

                let sim_a_gamma = sum_sim_a.powf(parms.gamma);
                let b1_sim_a_gamma = parms.b[0] * sim_a_gamma;

                b1_sim_a_gamma / (b1_sim_a_gamma + (parms.b[1] * (sum_sim_b.powf(parms.gamma))))
            })
            .collect();

        peffective
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_works() {
        let data = import_datum_from_csv("./data.csv");
        println!("{:?}", data);

        let gcm = GCMCompetent::new(data);
        let ans = gcm.predict(&Params {
            c: 8.3642,
            gamma: 3.8130,
            w: [6.2592, 1.6000, 1.1773, 1.6842, 1.4307, 1.2353],
            b: [1.1773, 1.1773],
        });

        println!("pef -> {:#?}", ans);

        assert!(
            ans == vec![
                0.6566253201438202,
                0.07027382739484003,
                0.17074534153798923,
                0.21838173609847855,
                0.10919943566652632,
                0.4864479062036666,
                0.08151513893081123,
                0.044749378014224794,
                0.5859726667648621,
                0.16407983296116913,
                0.009483440737565458,
                0.2770161872803142,
                0.11751702531748456,
                0.057410375832254966,
                0.0286837092748452,
                0.10000911378526983,
                0.014217033501507279,
                0.0619559966880805,
                0.010793808337613203,
                0.018588814843270744,
                0.14995005645425266,
                0.20966571964358785,
                0.01199196009605466,
                0.10693170272826036,
                0.19285379811021916,
                0.11108389810460345,
                0.009635715723361438,
                0.10999321930267499,
                0.017243946657103378,
                0.09709854776127945,
                0.08253170860262168,
                0.08475623915338634,
                0.15788108403720238,
                0.1132317937127596,
                0.049981992933727314,
                0.08242363697532,
                0.10053786321345883,
                0.5666641865760109,
                0.041144094382699906,
                0.10286777186913815,
                0.0523725119326643,
                0.0847209816619901,
                0.04788251811017914,
                0.10364926491357414,
                0.6934446017198798,
                0.11391942814447621,
                0.052585187764739444,
                0.12063727250229614,
                0.18790957391214758,
                0.13574380265348518,
                0.40772901390828514,
                0.047821883815039504,
                0.030351848560607415,
                0.29111470628435526,
                0.18491305935128846,
                0.48941241933915114,
                0.06680253119820233,
                0.24213999063974756,
                0.028182687153036356,
                0.07285882570037588,
                0.24047464255195333,
                0.22582302970654738,
                0.1359664270345951,
                0.09532219490105778,
                0.016116222823917038,
                0.2489212850253632,
                0.06743089675363388,
                0.09729626361141815,
                0.1479520065565344,
                0.15039226905992348,
                0.17227274781810578,
                0.1655064713265181,
                0.08240983895844492,
                0.04166316116596446,
                0.5742933683611741,
                0.49852166059604314,
                0.24961365955283144,
                0.07246329009710592,
                0.1670166979870326,
                0.1271921007125497,
            ]
        );
    }
}

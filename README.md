Ligand Selection with Active Learning
---

### Results and visualizations
- Seed dataset
  - [Ligand inventory](http://143.198.128.149:8123/) (raw [csv](./workplace/inventory/inventory.csv))
  - [Dimensionality reduction of the ligand pool](http://143.198.128.149:8125/) with [selected features](./workplace/ligand_descriptors/calculate.py)
  - [One-ligand experiments](./workplace/one_ligand/c_vs_fom)
  - [Sampling](./workplace/sampler)
- One-ligand system predictions
  - [Suggestions](./workplace/one_ligand/output/suggestions.csv)
  - [Predictions from the suggester](http://143.198.128.149:8124/)
  - [Suggester cross validation](./workplace/one_ligand/vis_cv/)
  - [Suggester perdictions: leave one ligand out](./workplace/one_ligand/output/lolo.png)
export type Ranking = {
  algo: string;
  n: number;
  mean: number;
  se: number;
  median: number;
  vs_baseline_pct: number | null;
};

export type PairedTest = {
  algo: string;
  n: number;
  wins: number;
  losses: number;
  mean_adv: number;
  p: number;
  sig: boolean;
};

export type RegretCurve = {
  algo: string;
  x: number[];
  mean: number[];
  se: number[];
  n: number;
};

export type SplitStats = {
  [split: string]: { [algo: string]: number };
};

export type LiveResponse = {
  experiment: {
    run_id: string;
    T: number | null;
    n_seeds: number | null;
    n_configs: number | null;
    total_trials: number;
    completed_trials: number;
    started_at: string | null;
    model: string;
    variants: string[];
  };
  stats: {
    rankings: Ranking[];
    paired: PairedTest[];
    curves: RegretCurve[];
    splits: SplitStats;
  };
  updated_at: number;
};

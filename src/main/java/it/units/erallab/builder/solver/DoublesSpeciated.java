package it.units.erallab.builder.solver;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.solver.Individual;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.StopConditions;
import it.units.malelab.jgea.core.solver.speciation.KMeansSpeciator;
import it.units.malelab.jgea.core.solver.speciation.SpeciatedEvolver;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.distance.LNorm;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.UniformCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public class DoublesSpeciated implements NamedProvider<SolverBuilder<List<Double>>> {
  private final double xOverProb;
  private final double sigmaMut;
  private final double rankBase;
  private final Function<?, double[]> converter;

  public DoublesSpeciated(
      double xOverProb,
      double sigmaMut,
      double rankBase,
      Function<?, double[]> converter
  ) {
    this.xOverProb = xOverProb;
    this.sigmaMut = sigmaMut;
    this.rankBase = rankBase;
    this.converter = converter;
  }

  @Override
  public SolverBuilder<List<Double>> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean remap = Boolean.parseBoolean(params.getOrDefault("remap", "false"));
    int nSpecies = Integer.parseInt(params.get("nSpecies"));
    return new SolverBuilder<>() {
      @SuppressWarnings("unchecked")
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<List<Double>, S, Q>,
          TotalOrderQualityBasedProblem<S, Q>, S> build(
          PrototypedFunctionBuilder<List<Double>, S> builder, S target
      ) {
        IndependentFactory<List<Double>> doublesFactory = new FixedLengthListFactory<>(builder.exampleFor(target)
            .size(), new UniformDoubleFactory(-1d, 1d));
        Map<GeneticOperator<List<Double>>, Double> geneticOperators = Map.of(
            new GaussianMutation(sigmaMut), 1d - xOverProb,
            new UniformCrossover<>(doublesFactory).andThen(new GaussianMutation(sigmaMut)), xOverProb
        );
        return new SpeciatedEvolver<>(
            builder.buildFor(target),
            doublesFactory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            remap,
            nPop / nSpecies,
            new KMeansSpeciator<>(
                nSpecies,
                -1,
                new LNorm(2),
                (Function<Individual<List<Double>, S, Q>, double[]>) converter
            ),
            rankBase
        );
      }
    };
  }
}

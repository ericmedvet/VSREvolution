package it.units.erallab.builder.solver;

import com.google.common.collect.Range;
import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.solver.SolverBuilder;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.operator.Crossover;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.StandardEvolver;
import it.units.malelab.jgea.core.solver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.solver.StopConditions;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.UniformCrossover;
import it.units.malelab.jgea.representation.sequence.bit.BitFlipMutation;
import it.units.malelab.jgea.representation.sequence.bit.BitString;
import it.units.malelab.jgea.representation.sequence.bit.BitStringFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;

/**
 * @author eric
 */
public class BinaryAndDoublesBiased implements NamedProvider<SolverBuilder<Pair<BitString, List<Double>>>> {

  private final double xOverProb;
  private final double tournamentRate;
  private final int minNTournament;

  public BinaryAndDoublesBiased(double xOverProb, double tournamentRate, int minNTournament, double pMut) {
    this.xOverProb = xOverProb;
    this.tournamentRate = tournamentRate;
    this.minNTournament = minNTournament;
  }

  @Override
  public SolverBuilder<Pair<BitString, List<Double>>> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean diversity = Boolean.parseBoolean(params.getOrDefault("diversity", "false"));
    boolean remap = Boolean.parseBoolean(params.getOrDefault("remap", "false"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<Pair<BitString, List<Double>>, S, Q>, TotalOrderQualityBasedProblem<S, Q>, S> build(PrototypedFunctionBuilder<Pair<BitString, List<Double>>, S> builder, S target) {
        Pair<BitString, List<Double>> sampleGenotype = builder.exampleFor(target);
        int bitStringLength = sampleGenotype.first().size();
        IndependentFactory<BitString> biasedFactory = random -> {
          BitString bitString = new BitString(bitStringLength);
          for (int i = 0; i < bitStringLength / 3; i++) {
            bitString.set(i, false);
          }
          for (int i = bitStringLength / 3; i < bitStringLength; i++) {
            bitString.set(i, random.nextBoolean());
          }
          return bitString;
        };
        int doublesLength = sampleGenotype.second().size();
        Factory<Pair<BitString, List<Double>>> factory = Factory.pair(
            biasedFactory,
            new FixedLengthListFactory<>(doublesLength, new UniformDoubleFactory(-1d, 1d)));
        Map<GeneticOperator<Pair<BitString, List<Double>>>, Double> geneticOperators = Map.of(
            Mutation.pair(
                new BitFlipMutation(.01d),
                new GaussianMutation(.35d)
            ), 1d - xOverProb,
            Crossover.pair(
                new UniformCrossover<>(new BitStringFactory(bitStringLength)),
                new GeometricCrossover(Range.closed(-.5d, 1.5d))
            ).andThen(
                Mutation.pair(new BitFlipMutation(.01d), new GaussianMutation(.1d))
            ), xOverProb
        );
        if (!diversity) {
          return new StandardEvolver<>(
              builder.buildFor(target),
              factory,
              nPop,
              StopConditions.nOfFitnessEvaluations(nEval),
              geneticOperators,
              new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
              new Last(),
              nPop,
              true,
              remap,
              (p, r) -> new POSetPopulationState<>()
          );
        }
        return new StandardWithEnforcedDiversityEvolver<>(
            builder.buildFor(target),
            factory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>(),
            100
        );
      }
    };
  }

}

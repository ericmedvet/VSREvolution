package it.units.erallab.builder.solver;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.solver.representation.STDPLearningRuleFactory;
import it.units.erallab.builder.solver.representation.STDPLearningRuleMutation;
import it.units.erallab.builder.solver.SolverBuilder;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.STDPLearningRule;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.StandardEvolver;
import it.units.malelab.jgea.core.solver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.solver.StopConditions;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.ProbabilisticMutation;
import it.units.malelab.jgea.representation.sequence.SameTwoPointsCrossover;

import java.util.List;
import java.util.Map;

public class STDPStandard implements NamedProvider<SolverBuilder<List<STDPLearningRule>>> {

  private final double xOverProb;
  private final double tournamentRate;
  private final int minNTournament;

  public STDPStandard(double xOverProb, double tournamentRate, int minNTournament) {
    this.xOverProb = xOverProb;
    this.tournamentRate = tournamentRate;
    this.minNTournament = minNTournament;
  }

  @Override
  public SolverBuilder<List<STDPLearningRule>> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean diversity = Boolean.parseBoolean(params.getOrDefault("diversity", "false"));
    boolean remap = Boolean.parseBoolean(params.getOrDefault("remap", "false"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<List<STDPLearningRule>, S, Q>, TotalOrderQualityBasedProblem<S, Q>, S> build(PrototypedFunctionBuilder<List<STDPLearningRule>, S> builder, S target) {
        int length = builder.exampleFor(target).size();
        IndependentFactory<List<STDPLearningRule>> factory = new FixedLengthListFactory<>(length, new STDPLearningRuleFactory());
        Map<GeneticOperator<List<STDPLearningRule>>, Double> geneticOperators = Map.of(
            new ProbabilisticMutation<>(0.2, factory, new STDPLearningRuleMutation(0.5, new STDPLearningRuleFactory())), 1 - xOverProb,
            new SameTwoPointsCrossover<>(factory).andThen(new ProbabilisticMutation<>(0.2, factory, new STDPLearningRuleMutation(0.5, new STDPLearningRuleFactory()))), xOverProb
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

package it.units.erallab.builder.solver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public interface SolverBuilder<G> {
  <S, Q> IterativeSolver<? extends POSetPopulationState<G, S, Q>, TotalOrderQualityBasedProblem<S, Q>, S> build(
      PrototypedFunctionBuilder<G, S> builder,
      S target
  );

}

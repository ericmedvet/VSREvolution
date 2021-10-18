package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.SparseMutation;
import it.units.malelab.jgea.representation.sequence.numeric.SparseSymmetricDoubleFactory;

import java.util.List;
import java.util.Map;

public class DoublesInitiallySparse implements EvolverBuilder<List<Double>> {

  private final int nPop;
  private final int nTournament;
  private final double p;

  public DoublesInitiallySparse(int nPop, int nTournament, double p) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.p = p;
  }

  @Override
  public <T, F> Evolver<List<Double>, T, F> build(PrototypedFunctionBuilder<List<Double>, T> builder, T target, PartialComparator<F> comparator) {
    int length = builder.exampleFor(target).size();
    return new StandardEvolver<>(
            builder.buildFor(target),
            new FixedLengthListFactory<>(length, new SparseSymmetricDoubleFactory(p, 0.5, 1.0)),
            comparator.comparing(Individual::getFitness),
            nPop,
            Map.of(
                    new SparseMutation(p, 0.35, new SparseSymmetricDoubleFactory(1d, 0.5, 1d)), 1d
            ),
            new Tournament(nTournament),
            new Last(),
            nPop,
            true,
            true
    );
  }


}

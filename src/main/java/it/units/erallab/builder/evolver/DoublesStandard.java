package it.units.erallab.builder.evolver;

import com.google.common.collect.Range;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;

/**
 * @author eric
 */
public class DoublesStandard implements EvolverBuilder<List<Double>> {

  private final int nPop;
  private final int nTournament;
  private final double xOverProb;
  private final double sigmaMutation = 0.35;
  protected final boolean diversityEnforcement;
  private final boolean remap;

  public DoublesStandard(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement, boolean remap) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
    this.diversityEnforcement = diversityEnforcement;
    this.remap = remap;
  }

  @Override
  public <T, F> Evolver<List<Double>, T, F> build(PrototypedFunctionBuilder<List<Double>, T> builder, T target, PartialComparator<F> comparator) {
    int length = builder.exampleFor(target).size();
    if (!diversityEnforcement) {
      return new StandardEvolver<>(
          builder.buildFor(target),
          new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
          comparator.comparing(Evolver.Individual::fitness),
          nPop,
          Map.of(
              new GaussianMutation(sigmaMutation), 1d - xOverProb,
              new GeometricCrossover(Range.closed(-.5d, 1.5d)).andThen(new GaussianMutation(sigmaMutation)), xOverProb
          ),
          new Tournament(nTournament),
          new Last(),
          nPop,
          true,
          remap
      );
    }
    return new StandardWithEnforcedDiversityEvolver<>(
        builder.buildFor(target),
        new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
        comparator.comparing(Evolver.Individual::fitness),
        nPop,
        Map.of(
            new GaussianMutation(sigmaMutation), 1d - xOverProb,
            new GeometricCrossover(Range.closed(-.5d, 1.5d)).andThen(new GaussianMutation(sigmaMutation)), xOverProb
        ),
        new Tournament(nTournament),
        new Last(),
        nPop,
        true,
        remap,
        100
    );
  }

}

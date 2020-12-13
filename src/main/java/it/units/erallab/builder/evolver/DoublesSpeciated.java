package it.units.erallab.builder.evolver;

import com.google.common.collect.Range;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.SpeciatedEvolver;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.distance.Distance;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;

/**
 * @author eric
 */
public class DoublesSpeciated extends DoublesStandard {

  private final double dThreshold;

  public DoublesSpeciated(int nPop, int nTournament, double xOverProb, double dThreshold) {
    super(nPop, nTournament, xOverProb);
    this.dThreshold = dThreshold;
  }

  private static class EuclideanDistance implements Distance<List<Double>> {
    @Override
    public Double apply(List<Double> v1, List<Double> v2) {
      double sum = 0d;
      for (int i = 0; i < Math.min(v1.size(), v2.size()); i++) {
        sum = sum + Math.pow(v1.get(i) - v2.get(i), 2d);
      }
      return Math.sqrt(sum);
    }
  }

  @Override
  public <T, F> Evolver<List<Double>, T, F> build(PrototypedFunctionBuilder<List<Double>, T> builder, T target, PartialComparator<F> comparator) {
    int length = builder.exampleFor(target).size();
    EuclideanDistance d = new EuclideanDistance();
    return new SpeciatedEvolver<>(
        builder.buildFor(target),
        new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
        comparator.comparing(Individual::getFitness),
        nPop,
        Map.of(
            new GaussianMutation(1d), 1d - xOverProb,
            new GeometricCrossover(Range.closed(-.5d, 1.5d)), xOverProb
        ),
        nPop / 10,
        (i1, i2) -> d.apply(i1.getGenotype(), i2.getGenotype()),
        dThreshold,
        Misc::first,
        0.75d
    );
  }

}

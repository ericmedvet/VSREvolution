package it.units.erallab.mapper.evolver;

import it.units.erallab.mapper.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.CMAESEvolver;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;

/**
 * @author eric
 */
public class CMAES implements EvolverBuilder<List<Double>> {
  @Override
  public <T> Evolver<List<Double>, T, Double> build(PrototypedFunctionBuilder<List<Double>, T> builder, T target) {
    int length = builder.exampleFor(target).size();
    return new CMAESEvolver<>(
        builder.buildFor(target),
        new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
        PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness)
    );
  }
}

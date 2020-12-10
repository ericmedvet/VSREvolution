package it.units.erallab.mapper.evolver;

import com.google.common.collect.Range;
import it.units.erallab.mapper.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Worst;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public class DoublesStandard implements EvolverBuilder<List<Double>> {

  protected final int nPop;
  protected final int nTournament;
  protected final double xOverProb;

  public DoublesStandard(int nPop, int nTournament, double xOverProb) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
  }

  @Override
  public <T> Evolver<List<Double>, T, Double> build(PrototypedFunctionBuilder<List<Double>, T> builder, T target) {
    int length = builder.exampleFor(target).size();
    return new StandardEvolver<>(
        builder.buildFor(target),
        new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
        PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness),
        nPop,
        Map.of(
            new GaussianMutation(1d), 1d - xOverProb,
            new GeometricCrossover(Range.closed(-.5d, 1.5d)), xOverProb
        ),
        new Tournament(nTournament),
        new Worst(),
        nPop,
        true
    );
  }

}

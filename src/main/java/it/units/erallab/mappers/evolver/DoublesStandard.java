package it.units.erallab.mappers.evolver;

import com.google.common.collect.Range;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Utils;
import it.units.erallab.mappers.MLP;
import it.units.erallab.mappers.ReversableFunction;
import it.units.erallab.mappers.robot.*;
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
  public <T> Evolver<List<Double>, Robot<?>, Double> build(ReversableFunction<List<Double>, T> innerMapper, RobotMapper<T> outerMapper) {
    PartialComparator<Individual<?, Robot<?>, Double>> comparator = PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness);
    int length = innerMapper.example(outerMapper.example()).size();
    System.out.println(length);
    return new StandardEvolver<>(
        innerMapper.andThen(outerMapper),
        new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
        comparator,
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

  public static void main(String[] args) {
    for (int l = 5; l < 20; l++) {
      Grid<? extends SensingVoxel> body = Utils.buildBody(String.format("biped-%dx2-f-f", l));
      System.out.printf("l=%d%n", l);
      new DoublesStandard(100, 5, 0.75d).build(
          new MLP(0.65d),
          new PhaseFunction(body, 1d, 1d)
      );
      new DoublesStandard(100, 5, 0.75d).build(
          ReversableFunction.identity(),
          new PhaseValues(body, 1d, 1d)
      );
      new DoublesStandard(100, 5, 0.75d).build(
          new MLP(0.65d),
          new Centralized(body)
      );
    }
  }
}

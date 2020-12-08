package it.units.erallab.mappers.evolver;

import com.google.common.collect.Range;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Utils;
import it.units.erallab.mappers.ReversableMapper;
import it.units.erallab.mappers.robot.PhaseValues;
import it.units.erallab.mappers.robot.RobotMapper;
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
import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public class Standard<T> extends Doubles<T> {

  protected final int nPop;
  protected final int nTournament;
  protected final double xOverProb;

  public Standard(ReversableMapper<Grid<? extends SensingVoxel>, List<Double>, T> mapper, int nPop, int nTournament, double xOverProb) {
    super(mapper);
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
  }

  @Override
  public Evolver<List<Double>, Robot<?>, Double> apply(Grid<? extends SensingVoxel> body, RobotMapper<T> tRobotMapper) {
    PartialComparator<Individual<?, Robot<?>, Double>> comparator = PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness);
    Function<List<Double>, T> innerMapper = mapper.apply(body);
    Function<T, Robot<?>> outerMapper = tRobotMapper.apply(body);
    int length = mapper.example(body).size();
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
    RobotMapper<List<Double>> robotMapper = new PhaseValues(1d, 1d);
    EvolverMapper<List<Double>, List<Double>> evolverMapper = new Standard<>(
        ReversableMapper.identityOn(robotMapper),
        100,
        5,
        0.75d
    );
    Grid<? extends SensingVoxel> body = Utils.buildBody("biped-4x2-f-f");
    Evolver<List<Double>, Robot<?>, Double> evolver = evolverMapper.apply(body, robotMapper);
  }
}

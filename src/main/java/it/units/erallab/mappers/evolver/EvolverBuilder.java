package it.units.erallab.mappers.evolver;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.mappers.ReversableFunction;
import it.units.erallab.mappers.robot.RobotMapper;
import it.units.malelab.jgea.core.evolver.Evolver;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public interface EvolverBuilder<G> {
  <T> Evolver<G, Robot<?>, Double> build(ReversableFunction<G, T> innerMapper, RobotMapper<T> outerMapper);
}

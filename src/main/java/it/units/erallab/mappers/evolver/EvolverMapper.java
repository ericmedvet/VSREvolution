package it.units.erallab.mappers.evolver;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.mappers.robot.RobotMapper;
import it.units.malelab.jgea.core.evolver.Evolver;

import java.util.function.BiFunction;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public interface EvolverMapper<G, T> extends BiFunction<Grid<? extends SensingVoxel>, RobotMapper<T>, Evolver<G, Robot<?>, Double>> {
}

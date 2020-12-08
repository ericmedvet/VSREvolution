package it.units.erallab.mappers.robot;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.mappers.ReversableMapper;

import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public interface RobotMapper<T> extends ReversableMapper<Grid<? extends SensingVoxel>, T, Robot<?>> {
}

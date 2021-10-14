package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.tree.Tree;

import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.DoubleStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoConditionedTreeHomoMLP extends DevoTreeHomoMLP implements PrototypedFunctionBuilder<Pair<Tree<Double>, List<Double>>, UnaryOperator<Robot<? extends SensingVoxel>>> {

  private final Function<Voxel, Double> selectionFunction;
  private final boolean maxFirst;

  public DevoConditionedTreeHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, Function<Voxel, Double> selectionFunction, boolean maxFirst, int nInitial, int nStep) {
    super(innerLayerRatio, nOfInnerLayers, signals, nInitial, nStep);
    this.selectionFunction = selectionFunction;
    this.maxFirst = maxFirst;
  }

  public DevoConditionedTreeHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, Function<Voxel, Double> selectionFunction, int nInitial, int nStep) {
    this(innerLayerRatio, nOfInnerLayers, signals, selectionFunction, false, nInitial, nStep);
  }

  @Override
  protected Comparator<Tree<DecoratedValue>> getComparator(boolean reversed, Robot<? extends SensingVoxel> robot) {
    Comparator<Tree<DecoratedValue>> firstComparator = Comparator.comparing(t -> getParentPriority(t.content().x, t.content().y, robot));
    if (maxFirst) {
      firstComparator.reversed();
    }
    Comparator<Tree<DecoratedValue>> secondComparator = super.getComparator(reversed, robot);
    return firstComparator.thenComparing(secondComparator);
  }

  private double getParentPriority(int x, int y, Robot<? extends SensingVoxel> robot) {
    if (robot == null) {
      return 0d;
    }
    DoubleStream neighborsPriorities = EnumSet.allOf(Direction.class).stream()
        .map(d -> robot.getVoxels().get(x + d.deltaX, y + d.deltaY))
        .filter(Objects::nonNull)
        .mapToDouble(selectionFunction::apply);
    return maxFirst ? neighborsPriorities.max().orElse(0) : neighborsPriorities.min().orElse(0);
  }
}

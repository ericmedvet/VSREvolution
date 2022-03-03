package it.units.erallab.builder.devofunction;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.malelab.jgea.representation.tree.Tree;

import java.util.Comparator;
import java.util.EnumSet;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.DoubleStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoConditionedTreeHomoMLP extends DevoTreeHomoMLP {

  // TODO decide what to do with these
  private final Function<Voxel, Double> selectionFunction;
  private final boolean maxFirst;

  public DevoConditionedTreeHomoMLP(Function<Voxel, Double> selectionFunction, boolean maxFirst) {
    this.selectionFunction = selectionFunction;
    this.maxFirst = maxFirst;
  }

  @Override
  protected Comparator<Tree<DecoratedValue>> getComparator(boolean reversed, Robot robot) {
    Comparator<Tree<DecoratedValue>> firstComparator = Comparator.comparing(t -> getParentPriority(
        t.content().x,
        t.content().y,
        robot
    ));
    if (maxFirst) {
      firstComparator.reversed();
    }
    Comparator<Tree<DecoratedValue>> secondComparator = super.getComparator(reversed, robot);
    return firstComparator.thenComparing(secondComparator);
  }

  private double getParentPriority(int x, int y, Robot robot) {
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

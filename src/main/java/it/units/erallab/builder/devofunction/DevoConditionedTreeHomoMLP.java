package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.builder.robot.FixedHomoDistributed;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.tree.Tree;

import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoConditionedTreeHomoMLP implements PrototypedFunctionBuilder<Pair<Tree<Double>, List<Double>>, UnaryOperator<Robot<? extends SensingVoxel>>> {

  private static class DecoratedRobot extends Robot<SensingVoxel> {
    private final Tree<DecoratedValue> developmentTree;

    public DecoratedRobot(Controller<SensingVoxel> controller, Grid<? extends SensingVoxel> voxels, Tree<DecoratedValue> developmentTree) {
      super(controller, voxels);
      this.developmentTree = developmentTree;
    }

    public Tree<DecoratedValue> getDevelopmentTree() {
      return developmentTree;
    }
  }

  private static class DecoratedValue {
    int x;
    int y;
    double parentPriority;
    double value;
    boolean enabled;

    public DecoratedValue(int x, int y, double value, boolean enabled) {
      this.x = x;
      this.y = y;
      this.value = value;
      this.enabled = enabled;
    }

  }

  private enum Direction {
    N(0, 0, -1),
    E(1, 1, 0),
    S(2, 0, 1),
    W(3, -1, 0);
    public final int index;
    public final int deltaX;
    public final int deltaY;

    Direction(int index, int deltaX, int deltaY) {
      this.index = index;
      this.deltaX = deltaX;
      this.deltaY = deltaY;
    }
  }

  private final MLP mlp;
  private final FixedHomoDistributed fixedHomoDistributed;
  private final int nInitial;
  private final int nStep;

  private final Function<Voxel, Double> selectionFunction;
  private final boolean maxFirst;

  public DevoConditionedTreeHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, Function<Voxel, Double> selectionFunction, boolean maxFirst, int nInitial, int nStep) {
    mlp = new MLP(innerLayerRatio, nOfInnerLayers);
    fixedHomoDistributed = new FixedHomoDistributed(signals);
    this.selectionFunction = selectionFunction;
    this.maxFirst = maxFirst;
    this.nInitial = nInitial;
    this.nStep = nStep;
  }

  public DevoConditionedTreeHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, Function<Voxel, Double> selectionFunction, int nInitial, int nStep) {
    this(innerLayerRatio, nOfInnerLayers, signals, selectionFunction, false, nInitial, nStep);
  }

  @Override
  @SuppressWarnings({"unchecked", "rawtypes"})
  public Function<Pair<Tree<Double>, List<Double>>, UnaryOperator<Robot<? extends SensingVoxel>>> buildFor(UnaryOperator<Robot<? extends SensingVoxel>> robotUnaryOperator) {
    Robot<? extends SensingVoxel> target = robotUnaryOperator.apply(null);
    SensingVoxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
    return pair -> {
      Tree<Double> tree = pair.first();
      List<Double> list = pair.second();
      //check values size
      if (list.size() != mlpValuesSize) {
        throw new IllegalArgumentException(String.format(
            "Wrong values size: %d expected, %d found",
            mlpValuesSize, list.size()
        ));
      }
      return previous -> {
        int n;
        Tree<DecoratedValue> devoTree;
        if (previous == null) {
          devoTree = Tree.map(tree, v -> new DecoratedValue(0, 0, v, false));
          n = nInitial;
        } else {
          if (!(previous instanceof DecoratedRobot)) {
            throw new IllegalArgumentException("Previous robot is not decorated with devo tree; cannot develop");
          }
          devoTree = ((DecoratedRobot) previous).getDevelopmentTree();
          setParentPriorities(devoTree, previous);
          n = countEnabled(devoTree) + nStep;
        }
        develop(devoTree, n, true);
        int maxX = devoTree.topSubtrees().stream().mapToInt(t -> t.content().x).max().orElse(0);
        int maxY = devoTree.topSubtrees().stream().mapToInt(t -> t.content().y).max().orElse(0);
        Grid<Boolean> shape = Grid.create(maxX + 1, maxY + 1, false);
        devoTree.topSubtrees().stream()
            .filter(t -> t.content().enabled)
            .forEach(t -> shape.set(t.content().x, t.content().y, true));
        Grid<SensingVoxel> body = Grid.create(shape, b -> b ? SerializationUtils.clone(voxelPrototype) : null);
        if (body.values().stream().noneMatch(Objects::nonNull)) {
          body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
        }
        //build controller
        Robot<? extends SensingVoxel> robot = new Robot<>(Controller.empty(), body);
        TimedRealFunction timedRealFunction = mlp.buildFor(fixedHomoDistributed.exampleFor(target)).apply(list);
        return new DecoratedRobot(
            (Controller) fixedHomoDistributed.buildFor(robot).apply(timedRealFunction).getController(),
            robot.getVoxels(),
            devoTree
        );
      };
    };
  }

  @Override
  public Pair<Tree<Double>, List<Double>> exampleFor(UnaryOperator<Robot<? extends SensingVoxel>> robotUnaryOperator) {
    Robot<? extends SensingVoxel> target = robotUnaryOperator.apply(null);
    return Pair.of(
        Tree.of(0d),
        mlp.exampleFor(fixedHomoDistributed.exampleFor(target))
    );
  }

  private void setParentPriorities(Tree<DecoratedValue> devoTree, Robot<? extends SensingVoxel> previous) {
    devoTree.topSubtrees().stream().map(Tree::content).forEach(t -> {
          DoubleStream neighborsPriorities = EnumSet.allOf(Direction.class).stream()
              .map(d -> previous.getVoxels().get(t.x + d.deltaX, t.y + d.deltaY))
              .filter(Objects::nonNull)
              .mapToDouble(selectionFunction::apply);
          t.parentPriority = maxFirst ? neighborsPriorities.max().orElse(0) : neighborsPriorities.min().orElse(0);
        }
    );
  }

  private void develop(Tree<DecoratedValue> tree, int n, boolean reversed) {
    Comparator<Tree<DecoratedValue>> firstComparator = Comparator.comparing(t -> t.content().parentPriority);
    if (maxFirst) {
      firstComparator.reversed();
    }
    Comparator<Tree<DecoratedValue>> secondComparator = Comparator.comparingDouble(t -> t.content().value);
    if (reversed) {
      secondComparator = secondComparator.reversed();
    }
    Comparator<Tree<DecoratedValue>> comparator = firstComparator.thenComparing(secondComparator);
    decorate(tree);
    while (countEnabled(tree) < n) {
      List<Tree<DecoratedValue>> subtrees = tree.topSubtrees().stream()
          .filter(t -> (t.parent() == null) || (t.parent().content().enabled)) // consider only close to enabled
          .filter(t -> !t.content().enabled) // consider only not already enabled
          .filter( // consider only those for which there is not one already enabled
              tt -> tree.topSubtrees().stream()
                  .filter(t -> t.content().enabled)
                  .noneMatch(t -> t.content().x == tt.content().x && t.content().y == tt.content().y)
          )
          .sorted(comparator)
          .collect(Collectors.toList());
      if (subtrees.isEmpty()) {
        break;
      }
      subtrees.get(0).content().enabled = true;
    }
    //adjust coords
    int minX = tree.topSubtrees().stream().mapToInt(t -> t.content().x).min().orElse(0);
    int minY = tree.topSubtrees().stream().mapToInt(t -> t.content().y).min().orElse(0);
    tree.topSubtrees().forEach(t -> {
      t.content().x = t.content().x - minX;
      t.content().y = t.content().y - minY;
    });
  }

  private static void decorate(Tree<DecoratedValue> tree) {
    if (tree.parent() == null) {
      tree.content().x = 0;
      tree.content().y = 0;
    }
    if (tree.nChildren() > 0) {
      for (Direction d : Direction.values()) {
        tree.child(d.index).content().x = tree.content().x + d.deltaX;
        tree.child(d.index).content().y = tree.content().y + d.deltaY;
        decorate(tree.child(d.index));
      }
    }
  }

  private static int countEnabled(Tree<DecoratedValue> tree) {
    return tree.content().enabled ? (1 + tree.childStream().mapToInt(DevoConditionedTreeHomoMLP::countEnabled).sum()) : 0;
  }

}

package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.builder.robot.FixedHomoDistributed;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.tree.Tree;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoTreeHomoMLP implements PrototypedFunctionBuilder<Pair<Tree<Double>, List<Double>>, UnaryOperator<Robot<? extends SensingVoxel>>> {

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

  public DevoTreeHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, int nInitial, int nStep) {
    mlp = new MLP(innerLayerRatio, nOfInnerLayers);
    fixedHomoDistributed = new FixedHomoDistributed(signals);
    this.nInitial = nInitial;
    this.nStep = nStep;
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

  private static void develop(Tree<DecoratedValue> tree, int n, boolean reversed) {
    Comparator<Tree<DecoratedValue>> comparator = Comparator.comparingDouble(t -> t.content().value);
    if (reversed) {
      comparator = comparator.reversed();
    }
    while (countEnabled(tree) < n) {
      List<Tree<DecoratedValue>> subtrees = tree.topSubtrees().stream()
          .filter(t -> (t.parent() == null) || (t.parent().content().enabled))
          .filter(t -> !t.content().enabled)
          .sorted(comparator)
          .collect(Collectors.toList());
      if (subtrees.isEmpty()) {
        break;
      }
      Tree<DecoratedValue> selected = subtrees.get(0);
      selected.content().enabled = true;
      if (selected.parent() == null) {
        selected.content().x = 0;
        selected.content().y = 0;
      } else {
        for (Direction d : Direction.values()) {
          if (selected.parent().child(d.index) == selected) {
            selected.content().x = selected.parent().content().x + d.deltaX;
            selected.content().y = selected.parent().content().y + d.deltaY;
            break;
          }
        }
      }
    }
    //adjust coords
    int minX = tree.topSubtrees().stream().mapToInt(t -> t.content().x).min().orElse(0);
    int minY = tree.topSubtrees().stream().mapToInt(t -> t.content().y).min().orElse(0);
    tree.topSubtrees().forEach(t -> {
      t.content().x = t.content().x - minX;
      t.content().y = t.content().y - minY;
    });
  }

  private static int countEnabled(Tree<DecoratedValue> tree) {
    return tree.content().enabled ? (1 + tree.childStream().mapToInt(DevoTreeHomoMLP::countEnabled).sum()) : 0;
  }

}

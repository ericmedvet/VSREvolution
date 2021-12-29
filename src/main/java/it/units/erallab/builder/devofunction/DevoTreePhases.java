package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.tree.Tree;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoTreePhases implements PrototypedFunctionBuilder<Tree<Pair<Double, Double>>, UnaryOperator<Robot>> {

  private static class DecoratedRobot extends Robot {
    private final Tree<DecoratedValue> developmentTree;

    private DecoratedRobot(Controller controller, Grid<Voxel> voxels, Tree<DecoratedValue> developmentTree) {
      super(controller, voxels);
      this.developmentTree = developmentTree;
    }

  }

  private static class DecoratedValue {
    int x;
    int y;
    double value;
    double phase;
    boolean enabled;

    private DecoratedValue(int x, int y, double value, double phase, boolean enabled) {
      this.x = x;
      this.y = y;
      this.value = value;
      this.phase = phase;
      this.enabled = enabled;
    }
  }

  private enum Direction {
    N(0, 0, -1), E(1, 1, 0), S(2, 0, 1), W(3, -1, 0);
    private final int index;
    private final int deltaX;
    private final int deltaY;

    Direction(int index, int deltaX, int deltaY) {
      this.index = index;
      this.deltaX = deltaX;
      this.deltaY = deltaY;
    }
  }

  private final double frequency;
  private final double amplitude;
  protected final int nInitial;
  protected final int nStep;
  private final double controllerStep;

  public DevoTreePhases(double frequency, double amplitude, int nInitial, int nStep, double controllerStep) {
    this.frequency = frequency;
    this.amplitude = amplitude;
    this.nInitial = nInitial;
    this.nStep = nStep;
    this.controllerStep = controllerStep;
  }

  public DevoTreePhases(double frequency, double amplitude, int nInitial, int nStep) {
    this(frequency, amplitude, nInitial, nStep, 0d);
  }

  @Override
  public Function<Tree<Pair<Double, Double>>, UnaryOperator<Robot>> buildFor(UnaryOperator<Robot> robotUnaryOperator) {
    Robot target = robotUnaryOperator.apply(null);
    Voxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    return doubleTree -> previous -> {
      int n;
      Tree<DecoratedValue> devoTree;
      if (previous == null) {
        devoTree = Tree.map(doubleTree, p -> new DecoratedValue(0, 0, p.first(), p.second(), false));
        n = nInitial;
      } else {
        if (!(previous instanceof DecoratedRobot)) {
          throw new IllegalArgumentException("Previous robot is not decorated with devo tree; cannot develop");
        }
        devoTree = ((DecoratedRobot) previous).developmentTree;
        n = countEnabled(devoTree) + nStep;
      }
      Comparator<Tree<DecoratedValue>> contentComparator = Comparator.comparingDouble(t -> t.content().value);
      develop(devoTree, contentComparator.reversed(), n);
      int maxX = devoTree.topSubtrees().stream().mapToInt(t -> t.content().x).max().orElse(0);
      int maxY = devoTree.topSubtrees().stream().mapToInt(t -> t.content().y).max().orElse(0);
      Grid<Boolean> shape = Grid.create(maxX + 1, maxY + 1, false);
      Grid<Double> phases = Grid.create(maxX + 1, maxY + 1, 0d);
      devoTree.topSubtrees().stream().filter(t -> t.content().enabled).forEach(t -> {
        shape.set(t.content().x, t.content().y, true);
        phases.set(t.content().x, t.content().y, t.content().phase);
      });
      Grid<Voxel> body = Grid.create(shape, b -> b ? SerializationUtils.clone(voxelPrototype) : null);
      if (body.values().stream().noneMatch(Objects::nonNull)) {
        body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
      }
      //build controller
      Robot robot = new Robot(Controller.empty(), body);
      double localAmplitude = amplitude;
      double localFrequency = frequency;
      AbstractController controller = new TimeFunctions(Grid.create(phases.getW(),
          phases.getH(),
          (x, y) -> t -> localAmplitude * Math.sin(2 * Math.PI * localFrequency * t + phases.get(x, y))
      ));
      if (controllerStep > 0) {
        controller = controller.step(controllerStep);
      }
      return new DecoratedRobot(controller, robot.getVoxels(), devoTree);
    };
  }

  @Override
  public Tree<Pair<Double, Double>> exampleFor(UnaryOperator<Robot> robotUnaryOperator) {
    return Tree.of(Pair.of(0d, 0d));
  }

  private static void develop(Tree<DecoratedValue> tree, Comparator<Tree<DecoratedValue>> comparator, int n) {
    decorate(tree);
    while (countEnabled(tree) < n) {
      List<Tree<DecoratedValue>> subtrees = tree.topSubtrees()
          .stream()
          .filter(t -> (t.parent() == null) || (t.parent().content().enabled)) // consider only close to enabled
          .filter(t -> !t.content().enabled) // consider only not already enabled
          .filter( // consider only those for which there is not one already enabled
              tt -> tree.topSubtrees()
                  .stream()
                  .filter(t -> t.content().enabled)
                  .noneMatch(t -> t.content().x == tt.content().x && t.content().y == tt.content().y))
          .sorted(comparator).toList();
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

  private Comparator<Tree<DecoratedValue>> getComparator(boolean reversed) {
    Comparator<Tree<DecoratedValue>> comparator = Comparator.comparingDouble(t -> t.content().value);
    if (reversed) {
      comparator = comparator.reversed();
    }
    return comparator;
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
    return tree.content().enabled ? (1 + tree.childStream().mapToInt(DevoTreePhases::countEnabled).sum()) : 0;
  }

}

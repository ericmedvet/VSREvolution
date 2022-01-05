package it.units.erallab;

import it.units.erallab.builder.DirectNumbersGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.devofunction.DevoHomoMLP;
import it.units.erallab.builder.devofunction.DevoTreeHomoMLP;
import it.units.erallab.builder.robot.BodyAndSinusoidal;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.Utils;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.util.TextPlotter;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import it.units.malelab.jgea.representation.tree.RampedHalfAndHalf;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

/**
 * @author "Eric Medvet" on 2021/10/12 for VSREvolution
 */
public class BiasesFinder {

  private static final String FILLER = " ░▒▓█";

  private interface FactoryBuilder<T> {
    Factory<T> buildFor(T target);
  }

  private record ProtoPair<G>(PrototypedFunctionBuilder<G, Robot> first, FactoryBuilder<G> second) {

    public List<Robot> generate(Robot target, int n, Random random) {
      G geno = first().exampleFor(target);
      Factory<G> factory = second().buildFor(geno);
      Function<G, Robot> robotMapper = first().buildFor(target);
      return factory.build(n, random).stream()
          .map(robotMapper)
          .collect(Collectors.toList());
    }
  }

  public static void main(String[] args) {
    Random random = new Random(1);
    int n = 1000;
    int nTop = 3;
    int nBars = 10;
    int gridW = 10;
    int gridH = 10;
    String targetSensorConfigName = "uniform-t+a-0.01";
    Robot target = new Robot(
        Controller.empty(),
        RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape("box-" + gridW + "x" + gridH))
    );
    //set pairs
    Map<String, ProtoPair<?>> protoPairs = new TreeMap<>(Map.of(
        "gridConnected-8", new ProtoPair<>(
            robotMapper(new DevoHomoMLP(1, 1, 1, 8, 0,0d)),
            g -> new FixedLengthListFactory<>(g.size(), new UniformDoubleFactory(-1d, 1d))
        ),
        "tree-8", new ProtoPair<>(
            robotMapper(new DevoTreeHomoMLP(1, 1, 1, 8, 0)),
            g -> Factory.pair(
                new RampedHalfAndHalf<>(3, 4, d -> 4, new UniformDoubleFactory(0d, 1d), new UniformDoubleFactory(0d, 1d)),
                new FixedLengthListFactory<>(g.second().size(), new UniformDoubleFactory(-1d, 1d)
                )
            )
        ),
        "largestConnected-50", new ProtoPair<>(
            new BodyAndSinusoidal(1d, 1d, 50, Set.of(BodyAndSinusoidal.Component.PHASE)).compose(new DirectNumbersGrid()),
            g -> new FixedLengthListFactory<>(g.size(), new UniformDoubleFactory(-1d, 1d))
        )
    ));
    //create and count
    for (String name : protoPairs.keySet()) {
      System.out.printf("Generating %d shapes with %s%n", n, name);
      ProtoPair<?> protoPair = protoPairs.get(name);
      List<Grid<Boolean>> originalShapes = protoPair.generate(target, n, random).stream()
          .map(r -> Grid.create(r.getVoxels(), Objects::nonNull)).toList();
      int maxW = originalShapes.stream()
          .map(s -> Utils.cropGrid(s, v -> v))
          .mapToInt(Grid::getW)
          .max()
          .orElse(1);
      int maxH = originalShapes.stream()
          .map(s -> Utils.cropGrid(s, v -> v))
          .mapToInt(Grid::getH)
          .max()
          .orElse(1);
      List<Grid<Boolean>> shapes = originalShapes.stream()
          .map(s -> translateAndCrop(s, maxW, maxH)).toList();
      //descriptors
      descriptors().forEach(descriptor -> {
        List<? extends Number> values = shapes.stream().map(descriptor).collect(Collectors.toList());
        System.out.printf(
            "%s (min=" + descriptor.getFormat() + ", avg=" + descriptor.getFormat() + ", max=" + descriptor.getFormat() + "): %s%n",
            descriptor.getName(),
            values.stream().mapToDouble(Number::doubleValue).min().orElse(Double.NaN),
            values.stream().mapToDouble(Number::doubleValue).average().orElse(Double.NaN),
            values.stream().mapToDouble(Number::doubleValue).max().orElse(Double.NaN),
            TextPlotter.histogram(values, nBars)
        );
      });
      //show density
      System.out.println("Density grid:");
      Grid<Double> density = Grid.create(
          maxW, maxH,
          (x, y) -> (double) shapes.stream().filter(s -> s.get(x, y)).count() / (double) shapes.size()
      );
      System.out.println(Grid.toString(
          density,
          (Function<Double, Character>) d -> FILLER.charAt(Math.round(d.floatValue() * (float) (FILLER.length() - 1)))
      ));
      //show top n most frequent
      Map<Grid<Boolean>, Integer> uniqueShapes = new HashMap<>();
      shapes.forEach(s -> uniqueShapes.put(s, 1 + uniqueShapes.getOrDefault(s, 0)));
      System.out.printf("Built %d on %d unique shapes%n", uniqueShapes.size(), shapes.size());
      List<Map.Entry<Grid<Boolean>, Integer>> entries = uniqueShapes.entrySet().stream()
          .sorted((e1, e2) -> Integer.compare(e2.getValue(), e1.getValue())).toList();
      System.out.printf("Hist of top %d shapes (min=%d, max=%d): %s%n",
          nBars,
          entries.get(0).getValue(),
          entries.get(Math.min(entries.size(), nBars) - 1).getValue(),
          TextPlotter.barplot(entries.stream().map(Map.Entry::getValue).limit(nBars).collect(Collectors.toList()))
      );
      for (int i = 0; i < Math.min(entries.size(), nTop); i++) {
        System.out.printf(
            "Most frequent shape n. %d (%d on %d occurrencies, %.4f):%n",
            i + 1, uniqueShapes.get(entries.get(i).getKey()), shapes.size(),
            (float) uniqueShapes.get(entries.get(i).getKey()) / (float) shapes.size()
        );
        System.out.println(Grid.toString(entries.get(i).getKey(), (Predicate<Boolean>) v -> v));
      }
    }
  }

  private static List<NamedFunction<Grid<Boolean>, ? extends Number>> descriptors() {
    return List.of(
        NamedFunction.build("nVoxels", "%.1f", s -> s.values().stream().filter(v -> v).count()),
        NamedFunction.build("w", "%.1f", s -> Utils.cropGrid(s, v -> v).getW()),
        NamedFunction.build("h", "%.1f", s -> Utils.cropGrid(s, v -> v).getH()),
        NamedFunction.build("center.x", "%.1f", s -> Utils.cropGrid(s, v -> v).stream()
            .filter(e -> e.value() != null)
            .mapToDouble(e -> (double) e.key().x() / (double) s.getW())
            .average().orElse(0d)),
        NamedFunction.build("center.y", "%.1f", s -> Utils.cropGrid(s, v -> v).stream()
            .filter(e -> e.value() != null)
            .mapToDouble(e -> (double) e.key().y() / (double) s.getH())
            .average().orElse(0d)),
        NamedFunction.build("compactness", "%.2f", Utils::shapeCompactness),
        NamedFunction.build("elongation", "%.2f", g -> Utils.shapeElongation(g,4)) // TODO check proper value for n
    );
  }

  private static <G> PrototypedFunctionBuilder<G, Robot> robotMapper(PrototypedFunctionBuilder<G, UnaryOperator<Robot>> devoFunction) {
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<G, Robot> buildFor(Robot robot) {
        UnaryOperator<Robot> targetDevoFunction = r -> (Robot) robot;
        return g -> devoFunction.buildFor(targetDevoFunction).apply(g).apply(null);
      }

      @Override
      public G exampleFor(Robot robot) {
        UnaryOperator<Robot> targetDevoFunction = r -> (Robot) robot;
        return devoFunction.exampleFor(targetDevoFunction);
      }
    };
  }

  private static Grid<Boolean> translateAndCrop(Grid<Boolean> grid, int w, int h) {
    int minX = grid.stream()
        .filter(Grid.Entry::value)
        .mapToInt(e -> e.key().x())
        .min()
        .orElse(0);
    int minY = grid.stream()
        .filter(Grid.Entry::value)
        .mapToInt(e -> e.key().y())
        .min()
        .orElse(0);
    return Grid.create(
        w, h,
        (x, y) -> isValid(grid, x + minX, y + minY) ? grid.get(x + minX, y + minY) : false
    );
  }

  private static boolean isValid(Grid<?> grid, int x, int y) {
    return x >= 0 && x < grid.getW() && y >= 0 && y < grid.getH();
  }
}

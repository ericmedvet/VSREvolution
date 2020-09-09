/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.BreakableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.malelab.jgea.core.util.Pair;
import org.apache.commons.lang3.SerializationUtils;

import java.io.*;
import java.util.*;
import java.util.function.UnaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * @author eric
 * @created 2020/08/19
 * @project VSREvolution
 */
public class Utils {

  private static final Logger L = Logger.getLogger(Utils.class.getName());

  private Utils() {
  }

  public static String safelySerialize(Serializable object) {
    try (
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(baos, true))
    ) {
      oos.writeObject(object);
      oos.flush();
      oos.close();
      return Base64.getEncoder().encodeToString(baos.toByteArray());
    } catch (IOException e) {
      L.log(Level.SEVERE, String.format("Cannot serialize due to %s", e), e);
      return "";
    }
  }

  public static <T> T safelyDeserialize(String string, Class<T> tClass) {
    try (
        ByteArrayInputStream bais = new ByteArrayInputStream(Base64.getDecoder().decode(string));
        ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(bais))
    ) {
      Object o = ois.readObject();
      return (T) o;
    } catch (IOException | ClassNotFoundException e) {
      L.log(Level.SEVERE, String.format("Cannot deserialize due to %s", e), e);
      return null;
    }
  }

  public static UnaryOperator<Robot<?>> buildRobotTransformation(String name) {
    String areaBreakable = "areaBreak-(?<rate>\\d+(\\.\\d+)?)-(?<threshold>\\d+(\\.\\d+)?)-(?<seed>\\d+)";
    String timeBreakable = "timeBreak-(?<time>\\d+(\\.\\d+)?)-(?<seed>\\d+)";
    String identity = "identity";
    if (name.matches(identity)) {
      return UnaryOperator.identity();
    }
    if (name.matches(areaBreakable)) {
      double rate = Double.parseDouble(paramValue(areaBreakable, name, "rate"));
      double threshold = Double.parseDouble(paramValue(areaBreakable, name, "threshold"));
      Random random = new Random(Integer.parseInt(paramValue(areaBreakable, name, "seed")));
      return robot -> {
        Robot<?> transformed = new Robot<>(
            ((Robot<SensingVoxel>) robot).getController(),
            Grid.create(SerializationUtils.clone((Grid<SensingVoxel>) robot.getVoxels()), v -> v == null ? null : (random.nextDouble() > rate ? v : new BreakableVoxel(
                v.getSensors(),
                random,
                Map.of(
                    BreakableVoxel.ComponentType.ACTUATOR, Set.of(BreakableVoxel.MalfunctionType.FROZEN),
                    BreakableVoxel.ComponentType.SENSORS, Set.of(BreakableVoxel.MalfunctionType.ZERO)
                ),
                Map.of(BreakableVoxel.MalfunctionTrigger.AREA, threshold)
            )))
        );
        return transformed;
      };
    }
    if (name.matches(timeBreakable)) {
      double time = Double.parseDouble(paramValue(timeBreakable, name, "time"));
      Random random = new Random(Integer.parseInt(paramValue(timeBreakable, name, "seed")));
      return robot -> {
        List<Pair<Integer, Integer>> coords = robot.getVoxels().stream()
            .filter(e -> e.getValue() != null)
            .map(e -> Pair.of(e.getX(), e.getY()))
            .collect(Collectors.toList());
        Collections.shuffle(coords, random);
        Grid<SensingVoxel> body = SerializationUtils.clone((Grid<SensingVoxel>) robot.getVoxels());
        for (int i = 0; i < coords.size(); i++) {
          int x = coords.get(i).first();
          int y = coords.get(i).second();
          body.set(x, y, new BreakableVoxel(
              body.get(x, y).getSensors(),
              random,
              Map.of(
                  BreakableVoxel.ComponentType.ACTUATOR, Set.of(BreakableVoxel.MalfunctionType.FROZEN),
                  BreakableVoxel.ComponentType.SENSORS, Set.of(BreakableVoxel.MalfunctionType.ZERO)
              ),
              Map.of(BreakableVoxel.MalfunctionTrigger.TIME, time * ((double) (i + 1) / (double) coords.size()))
          ));
        }
        return new Robot<>(
            ((Robot<SensingVoxel>) robot).getController(),
            body
        );
      };
    }
    throw new IllegalArgumentException(String.format("Unknown body name: %s", name));
  }

  public static String paramValue(String pattern, String string, String paramName) {
    Matcher matcher = Pattern.compile(pattern).matcher(string);
    if (matcher.matches()) {
      return matcher.group(paramName);
    }
    throw new IllegalStateException(String.format("Param %s not found in %s with pattern %s", paramName, string, pattern));
  }

  public static <E> List<E> ofNonNull(E... es) {
    List<E> list = new ArrayList<>();
    for (E e : es) {
      if (e != null) {
        list.add(e);
      }
    }
    return list;
  }

}
